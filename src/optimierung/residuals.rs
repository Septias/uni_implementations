#![allow(unused)]

use crate::DrawResult;
use nalgebra::{Vector2, Vector3};
use plotters::prelude::*;
use plotters_canvas::CanvasBackend;
use web_sys::HtmlCanvasElement;

/// Draw power function f(x) = x^power.
pub fn draw(canvas: HtmlCanvasElement) -> DrawResult<()> {
    let backend = CanvasBackend::with_canvas_object(canvas).unwrap();
    let root = backend.into_drawing_area();
    let font: FontDesc = ("akaasha", 20.0).into();

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(format!("Residuals"), font)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..10f32, 0f32..10f32)?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    chart.draw_series(LineSeries::new(
        (0..=1000)
            .map(|x| x as f32 / 100.0)
            .map(|x| (x, 2.7_f32.powf(x * 0.2))),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}

fn vec2(x: f32, y: f32) -> Vector2<f32> {
    [x, y].into()
}

struct Residual {}
struct GaussNewton<const N: usize> {
    points: [Vector2<f32>; N],
    parameters: Vector2<f32>,
}

impl<const N: usize> GaussNewton<N> {
    fn new(points: [Vector2<f32>; N]) -> Self {
        Self {
            points,
            parameters: [0.0, 0.0].into(),
        }
    }

    fn step() {
        let residuals = (0..N).map(|n| {});
        //let jacobian = [];
    }
}
