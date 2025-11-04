import numpy as np
from PIL import Image
import io
import os
import base64
import requests
import json
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        self.vllm_model_name = os.getenv("VLLM_MODEL", "/workspace/projects/nutrition-table/runs/exp10_merge_test_repro_20251023_2035/exp10_joint_merge_test_repro_12/merged")
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8008/v1/chat/completions")
        self.system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            "You are a Vision Language Model specialized in product images. Detect nutrition tables."
        )

    def execute(self, inference_requests):
        responses = []

        for request in inference_requests:
            image_np = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()

            buffered = io.BytesIO()
            Image.fromarray(image_np).save(buffered, format="PNG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {
                "model": self.vllm_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                            {
                                "type": "text",
                                "text": "Detect the bounding box of the nutrition table.",
                            },
                        ],
                    },
                ],
            }

            try:
                response = requests.post(
                    self.vllm_endpoint,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=float(os.getenv("TIMEOUT_SECS", "60")),
                )
                response.raise_for_status()
                output_text = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                output_text = f"vLLM Error: {str(e)}"

            out_tensor = pb_utils.Tensor(
                "text_output",
                np.array([output_text.encode("utf-8")], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
