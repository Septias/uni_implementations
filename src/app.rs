use std::ops::Deref;
use test_project::functions::residuals;
use web_sys::HtmlCanvasElement;
use yew::prelude::*;

fn redraw(canvas: &NodeRef) {
    let canvas = canvas.cast::<HtmlCanvasElement>().unwrap();
    residuals::draw(canvas).unwrap();
    //web_sys::console::log_1(&"rerender".into());
}

#[function_component(App)]
pub fn app() -> Html {
    let reference = use_state(|| NodeRef::default()).deref().clone();
    let reference_clone = reference.clone();
    let reference_clone2 = reference.clone();
    
    {
        use_effect( move || {
            let canvas = reference_clone.cast::<HtmlCanvasElement>().unwrap();
            canvas.set_width(1200);
            canvas.set_height(600);
            redraw(&reference_clone);
            ||()
        });
    }

    let onclick = Callback::from(move |_| {
        redraw(&reference_clone2);
    });

    html! {
        <main>
            <canvas ref={reference.clone()} />
            <button {onclick}> {"redraw"} </button>
        </main>
    }
}
