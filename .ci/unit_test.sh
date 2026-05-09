# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r requirements.txt
pip install -r requirements_dev.txt

# Global variable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"
INTEGRATION_REPORT_DIR="${PROJECT_ROOT}/test_reports/integration_tests"
TITAN_VERSION="v0.2.2"
TITAN_DIR="${PROJECT_ROOT}/third_party/torchtitan"
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}


# Run torchtitan upstream unit tests (with NPU patches applied)
run_upstream_ut() {
    echo "Running torchtitan upstream unit tests..."


    # Ensure torchtitan_npu is installed (applies patches on import)
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    # Clone torchtitan source if not exists
    if [ ! -d "$TITAN_DIR" ]; then
        echo "Cloning torchtitan source..."
        mkdir -p third_party
        git clone --branch $TITAN_VERSION --depth 1 \
            https://gitcode.com/GitHub_Trending/to/torchtitan.git $TITAN_DIR
    fi

    # Create conftest.py in torchtitan test dir to ensure import torchtitan_npu before each test
    local titan_test_dir="${TITAN_DIR}/tests/unit_tests"
    local conftest_file="${titan_test_dir}/conftest.py"

    if [[ ! -d "$titan_test_dir" ]]; then
        echo "Torchtitan unit test directory not found: $titan_test_dir"
        return 1
    fi
    cat > "$conftest_file" << 'EOF'
# Auto-generated conftest for torchtitan-npu patch testing
import pytest

def pytest_configure(config):
    """Import torchtitan_npu to apply NPU patches before running tests."""
    import torchtitan_npu  # noqa: F401
EOF

    # Save original PYTHONPATH and set torchtitan source path
    local saved_pythonpath="$PYTHONPATH"
    export PYTHONPATH="${TITAN_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

    pytest_args="-v --tb=short --import-mode=importlib"
    # Ignore tests incompatible with NPU environment (ut runs off-device)
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_tokenizer.py"
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_activation_checkpoint.py"
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_download_hf_assets.py"

    # Test target: torchtitan upstream unit tests
    local test_target="tests/unit_tests/"

    # Switch to torchtitan directory (tests use relative paths like ./torchtitan/models/...)
    cd "${TITAN_DIR}"
    echo "Running torchtitan tests from: $(pwd)"
    set +e
    python3 -m pytest $pytest_args $test_target
    local exit_code=$?
    set -e

    # Return to project root
    cd "$PROJECT_ROOT"

    # Cleanup: remove the generated conftest file
    rm -f "$conftest_file"

    # Restore PYTHONPATH
    export PYTHONPATH="$saved_pythonpath"

    if [[ $exit_code -eq 0 ]]; then
        echo "Torchtitan upstream tests passed!"
    elif [[ $exit_code -eq 5 ]]; then
        echo "No torchtitan tests found to run."
    else
        echo "Torchtitan upstream tests failed (exit_code=$exit_code)"
        exit $exit_code
    fi
}



# CI tests

run_upstream_ut
pytest -v --tb=short tests/unit_tests
