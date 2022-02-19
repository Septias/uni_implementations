pub mod functions;

/// Type alias for the result of a drawing function.
pub type DrawResult<T> = Result<T, Box<dyn std::error::Error>>;