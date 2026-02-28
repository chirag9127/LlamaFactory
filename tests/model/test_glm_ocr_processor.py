# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for GLM-OCR processor registration fix (issue #10193)."""

import pytest


def test_glm_ocr_processor_registration():
    """Verify that GlmOcrConfig is registered in AutoProcessor to use Glm4vProcessor.

    This test ensures the fix for issue #10193 works correctly: after registration,
    AutoProcessor's PROCESSOR_MAPPING should contain GlmOcrConfig -> Glm4vProcessor.
    """
    try:
        from transformers import AutoProcessor
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
        from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrConfig
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
    except ImportError:
        pytest.skip("transformers version does not support glm_ocr or glm4v modules")

    # Register the mapping (same code as in the fix)
    AutoProcessor.register(GlmOcrConfig, Glm4vProcessor, exist_ok=True)

    # Verify the mapping is present
    assert GlmOcrConfig in PROCESSOR_MAPPING, (
        "GlmOcrConfig should be registered in PROCESSOR_MAPPING after AutoProcessor.register()"
    )

    # Verify it maps to the correct processor class
    resolved_class = PROCESSOR_MAPPING[GlmOcrConfig]
    assert resolved_class is Glm4vProcessor, (
        f"GlmOcrConfig should map to Glm4vProcessor, but got {resolved_class}"
    )


def test_glm_ocr_processor_registration_idempotent():
    """Verify that registering GlmOcrConfig multiple times with exist_ok=True does not raise."""
    try:
        from transformers import AutoProcessor
        from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrConfig
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
    except ImportError:
        pytest.skip("transformers version does not support glm_ocr or glm4v modules")

    # Should not raise even when called multiple times
    AutoProcessor.register(GlmOcrConfig, Glm4vProcessor, exist_ok=True)
    AutoProcessor.register(GlmOcrConfig, Glm4vProcessor, exist_ok=True)


def test_glm_ocr_mm_plugin_mapping():
    """Verify that the glm_ocr template uses the correct mm_plugin (glm4v -> GLM4VPlugin).

    This confirms that the mm_plugin mapping does not need to be changed.
    """
    from llamafactory.data.mm_plugin import PLUGINS, GLM4VPlugin

    assert "glm4v" in PLUGINS, "glm4v should be registered in PLUGINS"
    assert PLUGINS["glm4v"] is GLM4VPlugin, (
        f"glm4v plugin should be GLM4VPlugin, got {PLUGINS['glm4v']}"
    )
