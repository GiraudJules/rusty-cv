mod bbox_bindings;
mod common;
mod image_bindings;
mod layout_bindings;
mod mask_bindings;
mod preprocess_bindings;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn rusty_cv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    image_bindings::register(m)?;
    mask_bindings::register(m)?;
    layout_bindings::register(m)?;
    bbox_bindings::register(m)?;
    preprocess_bindings::register(m)?;
    Ok(())
}
