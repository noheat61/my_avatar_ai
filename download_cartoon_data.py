google_drive_paths = {

    # generator
    # "KOREAN.pt" : "https://drive.google.com/uc?id=1B3r2tcI6DtvyJvI08Zk-G52AiDiirW19",
    # "KOREAN_encoder.pt" : "https://drive.google.com/uc?id=1XtsZLZu5Lq8-IAx8_96zFn0DUEQak71o",
    # "ASIAN.pt" : "https://drive.google.com/uc?id=1weP8MmXRB1Z55teH4Ex8e67VNCkK3Fcj",
    # "ASIAN_encoder.pt" : "https://drive.google.com/uc?id=11uby6Cr_hqzlNnWmIGXHExyI7uVh6sc4",
    "AMERICAN.pt" : "https://drive.google.com/uc?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO",
    "AMERICAN_encoder.pt" : "https://drive.google.com/uc?id=1QQuZGtHgD24Dn5E21Z2Ik25EPng58MoU",

    # cartoon
    # "KOREAN_DISNEY.pt" : "https://drive.google.com/uc?id=1tfDPAh4i9-bPldE6on5cv0xOD_jyWxx0",
    # "ASIAN_DISNEY.pt" : "https://drive.google.com/uc?id=1C9VJp2dHI7RRyCXJ-94n6J4-rffRVvbM",
    "AMERICAN_DISNEY.pt" : "https://drive.google.com/uc?id=1PILW-H4Q0W8S22TO4auln1Wgz8cyroH6",
    "AMERICAN_여신강림.pt" : "https://drive.google.com/uc?id=1yEky49SnkBqPhdWvSAwgK5Sbrc3ctz1y",
}

def download_pretrained_model(download_all=True, file=''):
    
    import os
    if not os.path.isdir('CartoonStyleGAN/networks'):
        os.makedirs('CartoonStyleGAN/networks')

    from gdown import download as drive_download
    
    if download_all==True:
        for nn in google_drive_paths:
            url = google_drive_paths[nn]
            networkfile = os.path.join('CartoonStyleGAN/networks', nn)
            drive_download(url, networkfile, quiet=False)

    else:
        url = google_drive_paths[file]
        networkfile = os.path.join('CartoonStyleGAN/networks', file)

        drive_download(url, networkfile, quiet=False)

if __name__ == "__main__":
    download_pretrained_model(True)