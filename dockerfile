FROM tensorflow/tensorflow:2.4.1-gpu

COPY upstride /opt/upstride/upstride
COPY setup.py /opt/upstride/setup.py
COPY version /opt/upstride/version
COPY README.md /opt/upstride/README.md
RUN cd /opt/upstride && \
    pip install . && \
    cd / && \
    rm -r /opt/upstride
