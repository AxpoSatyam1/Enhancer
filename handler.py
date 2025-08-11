import runpod
import base64
import tempfile
import os
from appformer.app import inference_app


def handler(event):
    """
    RunPod handler for CodeFormer with Base64 input and output.
    Expects:
    
        event['input']['image'] → Base64 string of the input image
    Returns:
        { "output_image": "<Base64 of processed image>" }
    """

    # 1️⃣ Get Base64 input
    image_b64 = event["input"].get("image")
    if not image_b64:
        return {"error": "No image provided"}

    # 2️⃣ Decode Base64 → Temp file
    image_data = base64.b64decode(image_b64)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        temp_img.write(image_data)
        temp_img_path = temp_img.name

    # 3️⃣ Run CodeFormer
    result = inference_app(
        image=temp_img_path,
        background_enhance=True,
        face_upsample=True,
        upscale=8,
        codeformer_fidelity=0.9
    )

    # 4️⃣ Extract output path
    output_path = None
    if isinstance(result, dict):
        output_path = result.get("output")
    elif isinstance(result, str):
        output_path = result

    if not output_path or not os.path.exists(output_path):
        return {"error": "No valid output from CodeFormer"}

    # 5️⃣ Read output image → Base64
    with open(output_path, "rb") as f:
        output_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 6️⃣ Cleanup temp files
    os.remove(temp_img_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    # 7️⃣ Return Base64 output
    return {"output_image": output_b64}


# 🚀 Start RunPod serverless
runpod.serverless.start({"handler": handler})
