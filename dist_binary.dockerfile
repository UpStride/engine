FROM tensorflow/tensorflow:2.2.0-gpu

COPY upstride /opt/upstride/upstride
COPY setup.py /opt/upstride/setup.py
RUN pip3 install --no-cache-dir Cython
RUN cd /opt/upstride && \
    python setup.py bdist_wheel
RUN python install --no-cache-dir dist/*.whl
RUN cd / && \
    rm -r /opt/upstride
