def plot4(imgs):
    for index, image in enumerate(imgs):
        plt.subplot(2, 2, index + 1)
        plt.axis('off')
        plt.imshow(image)
        plt.show()
    return None

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    return image
