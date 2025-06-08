use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// Helper function to find the project root and then the target binary
fn get_binary_path() -> Result<std::path::PathBuf, String> {
    // Assuming 'cargo test' is run from the workspace root or similar
    // Adjust if your test execution environment is different.
    let output = Command::new(env!("CARGO"))
        .arg("build")
        .arg("--bin")
        .arg("runtime_main")
        .arg("--message-format=json")
        .output()
        .map_err(|e| format!("Failed to execute cargo build: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Cargo build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let reader = BufReader::new(output.stdout.as_slice());
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
            if json["reason"] == "compiler-artifact" && json["target"]["name"] == "runtime_main" {
                if let Some(executable) = json["executable"].as_str() {
                    return Ok(std::path::PathBuf::from(executable));
                }
            }
        }
    }
    Err("Could not find executable path from cargo build output".to_string())
}

#[test]
fn test_runtime_main_executes_successfully() {
    let binary_path = match get_binary_path() {
        Ok(path) => path,
        Err(e) => {
            panic!("Failed to get binary path: {}", e);
        }
    };

    let mut cmd = Command::new(binary_path);
    cmd.current_dir(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    );
    cmd.stdout(Stdio::piped()); // Capture stdout
    cmd.stderr(Stdio::piped()); // Capture stderr

    // Set environment variables if necessary, e.g., for tracing
    cmd.env("RUST_LOG", "info");
    // Ensure the mock compute backend is used if not specified by default in runtime's Cargo.toml
    // However, runtime's Cargo.toml already specifies features=["mock"] for compute.

    let mut child = cmd.spawn().expect("Failed to spawn runtime_main process");

    let stdout_captured = Arc::new(Mutex::new(String::new()));
    let stderr_captured = Arc::new(Mutex::new(String::new()));

    let stdout_handle = {
        let stdout = child.stdout.take().expect("Failed to capture stdout");
        let reader = BufReader::new(stdout);
        let captured_data = Arc::clone(&stdout_captured);
        thread::spawn(move || {
            for line in reader.lines() {
                if let Ok(line_str) = line {
                    let mut data = captured_data.lock().unwrap();
                    data.push_str(&line_str);
                    data.push('\n'); // Add newline as lines() strips it
                }
            }
        })
    };

    let stderr_handle = {
        let stderr = child.stderr.take().expect("Failed to capture stderr");
        let reader = BufReader::new(stderr);
        let captured_data = Arc::clone(&stderr_captured);
        thread::spawn(move || {
            for line in reader.lines() {
                if let Ok(line_str) = line {
                    let mut data = captured_data.lock().unwrap();
                    data.push_str(&line_str);
                    data.push('\n');
                }
            }
        })
    };

    // Wait for the process to complete, with a timeout
    let timeout = Duration::from_secs(30); // Adjust timeout as needed
    match child.wait_timeout_secs(timeout) {
        Ok(Some(status)) => {
            stdout_handle.join().expect("Stdout reader thread panicked");
            stderr_handle.join().expect("Stderr reader thread panicked");

            let stdout_output = stdout_captured.lock().unwrap();
            let stderr_output = stderr_captured.lock().unwrap();

            eprintln!(
                "--- runtime_main STDOUT ---
{}",
                *stdout_output
            );
            eprintln!(
                "--- runtime_main STDERR ---
{}",
                *stderr_output
            );

            assert!(
                status.success(),
                "runtime_main process exited with error: {:?}",
                status.code()
            );
            assert!(
                stdout_output.contains("Running in headless mode. No renderer will be used."),
                "Expected log output not found in stdout."
            );
        }
        Ok(None) => {
            child.kill().expect("Failed to kill timed-out process");
            panic!("runtime_main process timed out after {:?}", timeout);
        }
        Err(e) => {
            panic!("Failed to wait for runtime_main process: {}", e);
        }
    }
}

// A helper trait and impl to use wait_timeout_secs (not in std Command on all Rust versions)
trait ChildExt {
    fn wait_timeout_secs(
        &mut self,
        duration: Duration,
    ) -> std::io::Result<Option<std::process::ExitStatus>>;
}

impl ChildExt for std::process::Child {
    fn wait_timeout_secs(
        &mut self,
        duration: Duration,
    ) -> std::io::Result<Option<std::process::ExitStatus>> {
        // This is a simplified cross-platform way.
        // For a more robust solution, consider using platform-specific APIs or crates like `wait-timeout`.
        let start_time = std::time::Instant::now();
        loop {
            match self.try_wait()? {
                Some(status) => return Ok(Some(status)),
                None => {
                    if start_time.elapsed() > duration {
                        return Ok(None); // Timeout
                    }
                    thread::sleep(Duration::from_millis(50)); // Poll interval
                }
            }
        }
    }
}

// We need to add serde_json to dev-dependencies of runtime crate for get_binary_path
// And potentially wait-timeout or similar if the simplified ChildExt is not robust enough.
// For now, this test setup is a good start.
