TITLE = "Inst-Inpaint: Instructing to Remove Objects with Diffusion Models"

DESCRIPTION = """
<p style='text-align: center'>
    <a href='http://instinpaint.abyildirim.com' target='_blank'>Project Page</a> | 
    <a href='https://arxiv.org/abs/2304.03246' target='_blank'>Paper</a> | 
    <a href='https://github.com/abyildirim/inst-inpaint' target='_blank'>GitHub Repo</a> | 
</p>
<p style='text-align: center'>
    This demo demonstrates the Inst-Inpaint's abilities for instruction-based image inpainting.
</p>
"""

EXAMPLES = [
    ["gradio_app/examples/kite-boy.png", "Remove the colorful kite", True],
    ["gradio_app/examples/cat-car.jpg", "Remove the car", True],
    ["gradio_app/examples/bus-tree.jpg", "Remove the red bus", True],
    ["gradio_app/examples/cups.webp", "Remove the cup at the left", True],
    ["gradio_app/examples/woman-fantasy.jpg", "Remove the woman", True],
    ["gradio_app/examples/clock.png", "Remove the round clock at the center", True],
    ["gradio_app/examples/woman.png", "Remove the woman at the left", True],
    ["gradio_app/examples/men.png", "Remove the man at the right", True],
    ["gradio_app/examples/tree.png", "Remove the tree", True],
    ["gradio_app/examples/birds.png", "Remove the bird at the right of the bird", True]
]
