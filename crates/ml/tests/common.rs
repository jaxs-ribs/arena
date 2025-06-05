use compute::{ComputeBackend, MockCpu};
use std::sync::Arc;

pub fn mock_backend() -> Arc<dyn ComputeBackend> {
    Arc::new(MockCpu::default())
} 