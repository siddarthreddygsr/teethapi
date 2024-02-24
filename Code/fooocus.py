import requests

class RunpodAPI:
    def __init__(self, api_key):
        # self.base_url = "https://api.runpod.ai/v2/j6ctvdmtlle2lt/"
        self.base_url = "https://3y7wkrmlouh1fu-8888.proxy.runpod.net/"
        self.headers = {
            "Content-Type": "application/json"
        }

    def run_job(self, input_data):
        url = f"{self.base_url}run"
        data = {"input": input_data}
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def get_job_status(self, job_id):
        url = f"{self.base_url}status/{job_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()

# def main():
#     api_key = "PT9H5DJMNWJ1PQKJW98PTT99COHABVR5MLEG4WY2"
#     runpod_api = RunpodAPI(api_key)

#     input_data = {
#         "api_name": "v2/generation/image-prompt",
#         "input_image": "http://0x0.st/HnXJ.png",
#         "cn_image": "https://i.imgur.com/RKi2BAY.png",
#         "input_mask": "https://i.imgur.com/tC6DV8I.png",
#         "require_base64": True,
#     }

#     job_response = runpod_api.run_job(input_data)
#     print(job_response)

#     if "job_id" in job_response:
#         job_id = job_response["job_id"]
#         status_response = runpod_api.get_job_status(job_id)
#         print(status_response)

# if __name__ == "__main__":
#     main()
