use std::ops::Deref;
use test_project::functions::mandelbrot;
use web_sys::HtmlCanvasElement;
use yew::prelude::*;


#[function_component(App)]
pub fn app() -> Html {
    let reference = use_state(|| NodeRef::default()).deref().clone();

    let reference_clone = reference.clone();
    {
        use_effect( move || {
            let canvas = reference_clone.cast::<HtmlCanvasElement>().unwrap();
            mandelbrot::draw(canvas).unwrap();
            web_sys::console::log_1(&"rerender".into());
            ||()
        });
    }

    html! {
        <main>
            <canvas ref={reference.clone()} />
        </main>
    }
}
