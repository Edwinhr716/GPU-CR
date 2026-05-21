# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams

import socket
import time

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    # Create an LLM.
    import os
    model = os.environ.get("VLLM_MODEL", "/path/to/your/model")
    llm = LLM(model=model, enforce_eager=True)
    
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('0.0.0.0', 10000))
    print("UDP listening on port 10000...", flush=True)

    while True:
        data, addr = udp_socket.recvfrom(1024)
        print(f"Received data from {addr}: {data.decode()}")
    
        # calculate time in milliseconds
        start = time.time()
        outputs = llm.generate([data.decode()], sampling_params)
        end = time.time()
        print(f"Time taken for generation: {(end - start) * 1000:.2f} milliseconds")
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)

        udp_socket.sendto(generated_text.encode(), addr)


if __name__ == "__main__":
    main()
