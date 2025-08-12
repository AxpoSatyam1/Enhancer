import runpod
import base64
import tempfile
import os
import numpy as np
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
    if isinstance(image_data, (bytes, bytearray)):
        image_data = np.frombuffer(image_data, np.uint8)


    # 3️⃣ Run CodeFormer
    result = inference_app(
        image=image_data,
        background_enhance=True,
        face_upsample=True,
        upscale=8,
        codeformer_fidelity=0.9
    )

    # 4️⃣ Extract outpu

    # if not output_path or not os.path.exists(output_path):
    #     return {"error": "No valid output from CodeFormer"}

    # # 5️⃣ Read output image → Base64
    # with open(output_path, "rb") as f:
    #     output_b64 = base64.b64encode(f.read()).decode("utf-8")

    
    

    # 7️⃣ Return Base64 output
    return {"output_image": result}


# 🚀 Start RunPod serverless
runpod.serverless.start({"handler": handler})
