"""SiliconShield desktop GUI: native Qt window over a persistent worker.

The window stays responsive because model work happens in a separate worker
process reached over local pipes. Demonstration inputs live in
examples/gui-demo/. Router and assessment work stays on the CLI.
"""
