import PIL


def get_resizing_and_cropping_parameters(original_height, original_width,
                                         final_height, final_width,
                                         resize_scale, crop_h):
    """
    get the resize and crop paramters according to original h&w, final h&w and resize methods
    :param original_height: height of original image
    :param original_width: width of original image
    :param final_height: height of output image
    :param final_width: width of output image
    :param resize_scale: scale of resizing method
    :param crop_h: top crop pixels after resizing
    :return params: dict of resizing params
    """
    resize_dims = (int(original_width * resize_scale),
                   int(original_height * resize_scale))
    resized_width, resized_height = resize_dims

    crop_w = int(max(0, (resized_width - final_width) / 2))
    # Left, top, right, bottom crops.
    crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

    if resized_width != final_width:
        print('Zero padding left and right parts of the image.')
    if crop_h + final_height != resized_height:
        print('Zero padding bottom part of the image.')

    return {
        'scale_width': resize_scale,
        'scale_height': resize_scale,
        'resize_dims': resize_dims,
        'crop': crop,
    }


def resize_and_crop_image(img, resize_dims, crop):
    """
    Bilinear resizing followed by cropping
    :param img: PIL image
    :param resize_dims: resize_dims output by params dict
    :param crop: crop param output by params dict
    :return img: output image after resize and crop
    """
    img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img
