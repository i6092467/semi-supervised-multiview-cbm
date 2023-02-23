import numpy as np

import cv2

import os
import glob
import ntpath

import progressbar

# In BGR format
WHITE_BOUNDARY = ([250, 250, 250], [255, 255, 255])
TURQUOISE_BOUNDARY = ([180, 180, 0], [255, 255, 150])
TURQUOISE_BOUNDARY_2 = ([220, 220, 165], [230, 230, 185])
GOLDEN_BOUNDARY = ([50, 180, 180], [180, 255, 255])
GREEN_BOUNDARY = ([90, 170, 30], [150, 255, 120])
ORANGE_BOUNDARY = ([0, 145, 235], [25, 155, 255])
ORANGE_BOUNDARY_2 = ([40, 165, 225], [60, 180, 235])
BG_BOUNDARY = ([62, 48, 50], [64, 50, 52])
BG_BOUNDARY_2 = ([132, 127, 128], [136, 130, 130])
BG_BOUNDARY_3 = ([130, 126, 127], [135, 130, 132])

RES_WIDTH = 680
RES_HEIGHT = 512

CORR_TH = 0.75


def create_marker_mask(file, boundaries, display=True, save_mask_file=None, pad_kernel_size=[10], resize=False,
                       res_size=(RES_HEIGHT, RES_WIDTH), save_img_file=None):
    # Load the image
    image = cv2.imread(file)
    ht, wd, cc = image.shape

    if resize:
        # Create new image of desired size and color
        color = (0, 0, 0)
        result = np.full((res_size[0], res_size[1], cc), color, dtype=np.uint8)

        # Compute center offset
        if ht > res_size[0]:
            y_range_res = np.arange(0, res_size[0])
            yy = (ht - res_size[0]) // 2
            y_range_img = np.arange(yy, yy + res_size[0])
        else:
            yy = (res_size[0] - ht) // 2
            y_range_res = np.arange(yy, yy + ht)
            y_range_img = np.arange(0, ht)
        if wd > res_size[1]:
            x_range_res = np.arange(0, res_size[1])
            xx = (wd - res_size[1]) // 2
            x_range_img = np.arange(xx, xx + res_size[1])
        else:
            xx = (res_size[1] - wd) // 2
            x_range_res = np.arange(xx, xx + wd)
            x_range_img = np.arange(0, wd)

        result[y_range_res[0]:(y_range_res[-1]+1), x_range_res[0]:(x_range_res[-1]+1)] = \
            image[y_range_img[0]:(y_range_img[-1]+1), x_range_img[0]:(x_range_img[-1]+1)]
        image = result

        # Save resized image
        if save_img_file is not None:
            cv2.imwrite(save_img_file, image)

    # Go through different colour boundaries
    mask = None
    cnt = 0
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # Find the colors within the specified boundaries and apply the mask
        mask_ = cv2.inRange(image, lower, upper)

        # Now pad the mask by applying a filter
        kernel = np.ones((pad_kernel_size[cnt], pad_kernel_size[cnt]))
        mask_ = cv2.filter2D(mask_, -1, kernel)

        if mask is None:
            mask = mask_.copy()
        else:
            mask = cv2.bitwise_or(mask, mask_)

        cnt += 1
    mask = np.stack([mask] * 3, axis=2)

    # Save the mask
    if save_mask_file is not None:
        cv2.imwrite(save_mask_file, mask)

    # Show the images
    if display:
        cv2.imshow("images", np.hstack([image, mask]))
        cv2.waitKey(0)


def prepare_inpainting(image_dir, test_dir, mask_dir, template_dir, verbose=1):
    image_file_list = glob.glob(image_dir + '/*')

    if verbose > 0:
        cnt = 0
        bar = progressbar.ProgressBar(maxval=len(image_file_list))
        bar.start()

    for image_file in image_file_list:
        image_file_wo_ext = image_file.replace('.png', '')
        image_file_wo_ext = image_file_wo_ext.replace('.bmp', '')
        image_file_wo_ext = image_file_wo_ext.replace('.jpg', '')
        image_file_wo_ext = image_file_wo_ext.replace('.jpeg', '')

        remove_patterns(template_dir=template_dir,
                        image_file=image_file,
                        dest_file=os.path.join(test_dir, ntpath.basename(image_file_wo_ext + '.png')))

        create_marker_mask(file=os.path.join(test_dir, ntpath.basename(image_file_wo_ext + '.png')),
                           boundaries=[WHITE_BOUNDARY, GOLDEN_BOUNDARY, TURQUOISE_BOUNDARY, TURQUOISE_BOUNDARY_2,
                                       GREEN_BOUNDARY, ORANGE_BOUNDARY, ORANGE_BOUNDARY_2, BG_BOUNDARY, BG_BOUNDARY_2, BG_BOUNDARY_3],
                           save_mask_file=os.path.join(mask_dir, ntpath.basename(image_file_wo_ext + '.png')),
                           display=False, pad_kernel_size=[10, 10, 10, 10, 10, 10, 10, 30, 30, 30], resize=True,
                           save_img_file=os.path.join(test_dir, ntpath.basename(image_file_wo_ext + '.png')))

        # Remove colour bands
        image = cv2.imread(filename=os.path.join(test_dir, ntpath.basename(image_file_wo_ext + '.png')))
        image[:, 0:20, :] = np.array([0, 0, 0])
        image[:, -20:, :] = np.array([0, 0, 0])
        image[0:20, :, :] = np.array([0, 0, 0])
        image[-20:, :, :] = np.array([0, 0, 0])
        cv2.imwrite(os.path.join(test_dir, ntpath.basename(image_file_wo_ext + '.png')), image)

        if verbose > 0:
            bar.update(cnt + 1)
            cnt += 1


# Resizes an image and maintains the aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def remove_patterns(template_dir, image_file, dest_file):
    # Load original image, convert to grayscale
    original_image = cv2.imread(image_file)
    final = original_image.copy()
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    template_file_list = glob.glob(template_dir + '/*')
    for template_file in template_file_list:
        # Load template, convert to grayscale, perform canny edge detection
        template = cv2.imread(template_file)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]

        found = None

        # Dynamically rescale image for better template matching
        for scale in np.concatenate((np.linspace(0.5, 1.5, 20)[::-1], np.array([1.0]))):

            # Resize image to scale and keep track of ratio
            resized = cv2.resize(gray, dsize=(0,0), fx=scale, fy=scale)
            r = gray.shape[1] / float(resized.shape[1])

            # Stop if template image size is larger than resized image
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # Detect edges in resized image and apply template matching
            canny = cv2.Canny(resized, 50, 200)
            detected = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(detected)

            # Keep track of correlation value
            # Higher correlation means better match
            if found is None:
                if max_val > CORR_TH:
                    found = (max_val, max_loc, r)
            else:
                if max_val > found[0]:
                    found = (max_val, max_loc, r)
        if found is not None:
            # Compute coordinates of bounding box
            (_, max_loc, r) = found
            (start_x, start_y) = (int(max_loc[0] * r), int(max_loc[1] * r))
            (end_x, end_y) = (int((max_loc[0] + tW) * r), int((max_loc[1] + tH) * r))

            # Erase unwanted ROI (Fill ROI with white)
            cv2.rectangle(final, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)

    cv2.imwrite(dest_file, final)

