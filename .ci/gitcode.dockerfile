# FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu22.04_x86:9.0.0-beta.1-910b-py3.11
FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu22.04_arm:9.0.0-beta.1-910b-py3.11


RUN mkdir /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://repo.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

RUN pip3 install esdk-obs-python --trusted-host mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple

RUN pip3 install --no-cache-dir "https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.12.0.dev20260317%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl"
RUN pip3 install --no-cache-dir --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
RUN git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git /tmp/torchtitan \
    && git -C /tmp/torchtitan checkout ac13e536c84e7f6647b14fa9375c3c8a8a2b8578
RUN pip3 install --no-cache-dir -r /tmp/torchtitan/requirements.txt
RUN pip3 install --no-cache-dir -e /tmp/torchtitan
RUN git clone https://gitcode.com/Ascend/pytorch.git /tmp/pytorch_npu \
    && git -C /tmp/pytorch_npu checkout f9cbf1f179b59e75b915a72cfc3187f0aadfdea3
RUN bash -lc 'cd /tmp/pytorch_npu && bash ci/build.sh --python=3.11'
RUN pip3 install --upgrade /tmp/pytorch_npu/dist/torch_npu*.whl
RUN pip3 install --no-cache-dir \
    pybind11 \
    triton-ascend==3.2.0 \
    scipy \
    safetensors==0.7.0 \
    pytest==7.3.2 \
    pytest-cov \
    pre-commit \
    pyrefly==0.45.1 \
    transformers \
    einops \
    expecttest \
    tomli_w

COPY ./cluster_smoke_task.sh /home/cluster_smoke_task.sh
COPY ./upload.py /home/upload.py
