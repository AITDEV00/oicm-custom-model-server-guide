FROM lmsysorg/sglang:v0.5.9-cu129-amd64-runtime

EXPOSE 8080-9000

USER root

RUN <<eot
    set -ex

    # Security: add user runner
    useradd --uid 10000 runner

    # Create app folder
    mkdir /home/runner && chown runner /home/runner
    mkdir -p /app && chown runner /app

    # Security: disable root login
    passwd -l root
eot

USER runner
ENV HOME /home/runner
ENV PATH "$HOME/.local/bin:$PATH"

COPY --chown=runner startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

WORKDIR /app

ENTRYPOINT ["/app/startup.sh"]
