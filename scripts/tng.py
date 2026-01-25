import requests

def get(path, headers={"api-key":"2b74d8d0c8b07218c4edfe38e7b0340f"}, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open("/Users/andreacosta/Desktop/dev/python/master_thesis_project/data/url_files/"+filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r