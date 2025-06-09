pub fn new() -> Result<Self> {
    let event_loop = EventLoop::new().context("create event loop")?;
    let window = WindowBuilder::new()
        .with_title("Differentiable Physics")
        .with_maximized(true)
        .build(&event_loop)
        .context("failed to create window")?;

    let instance = wgpu::Instance::default();

    // ... existing code ...

    Ok(Self {
        // ... existing code ...
    })
} 