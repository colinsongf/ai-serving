From ai-cpu-notebook:base
COPY ../ai-serving /serving-framework
RUN /bin/bash -c "source /py3/bin/activate && \
    cd /serving-framework  && \
    python setup.py install && cd .. && \
    rm -rf /root/.cache/"

RUN ln -sf  /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV SERVING_DIR=/serving
RUN mkdir $SERVING_DIR
WORKDIR $SERVING_DIR
CMD /bin/bash -c  "cd $SERVING_DIR && bash start.sh"
