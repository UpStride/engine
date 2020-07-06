FROM tensorflow/tensorflow:2.2.0-gpu

COPY upstride /opt/upstride/upstride
COPY dist_binary_setup.py /opt/upstride/dist_binary_setup.py
RUN pip3 install --no-cache-dir Cython
RUN cd /opt/upstride && \
    python dist_binary_setup.py bdist_wheel
RUN pip3 install --no-cache-dir /opt/upstride/dist/*.whl
RUN cd / && \
    rm -r /opt/upstride