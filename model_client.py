"""StereoVLN Model Client

A drop-in replacement for the StereoVLN model that communicates via HTTP.
Use this in the `streamvln` environment (Python 3.9) with Habitat.
"""
import requests
import torch
from typing import List


class StereoVLNClient:
    """HTTP client that mimics the StereoVLN model interface."""

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url.rstrip('/')
        self._check_connection()

    def _check_connection(self):
        """Check if the server is available."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Connected to StereoVLN server: {response.json()}")
            else:
                raise ConnectionError(f"Server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to StereoVLN server at {self.server_url}: {e}")

    def eval(self):
        """Dummy eval() to match model interface."""
        pass

    def to(self, device):
        """Dummy to() to match model interface."""
        return self

    def inference(
        self,
        instruction: List[str],
        history_action: List[str],
        left_current_frame: torch.Tensor,   # [B, 1, 3, 448, 448]
        right_current_frame: torch.Tensor,  # [B, 1, 3, 448, 448]
        right_history_video: torch.Tensor,  # [B, history_num, 3, 448, 448]
        max_new_tokens: int = 128,
    ) -> dict:
        """Send inference request to the server."""
        payload = {
            'instruction': instruction,
            'history_action': history_action,
            'left_current_frame': left_current_frame.cpu().float().tolist(),
            'right_current_frame': right_current_frame.cpu().float().tolist(),
            'right_history_video': right_history_video.cpu().float().tolist(),
            'max_new_tokens': max_new_tokens,
        }

        response = requests.post(
            f"{self.server_url}/inference",
            json=payload,
            timeout=120  # 2 minutes timeout for inference
        )

        if response.status_code != 200:
            error_msg = response.json().get('error', 'Unknown error')
            raise RuntimeError(f"Inference failed: {error_msg}")

        result = response.json()
        return {
            'generated_text': result['generated_text'],
            'left_point': torch.tensor(result['left_point']) if result.get('left_point') else None,
            'right_point': torch.tensor(result['right_point']) if result.get('right_point') else None,
        }
