from PIL import Image
import random
import numpy as np
import math


# 随机打乱像素
def pixelRandomChange(base_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    pixels = list(base_image.getdata())
    for _ in range(width * height):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(0, width - 1)
        y2 = random.randint(0, height - 1)
        pos1 = y1 * width + x1
        pos2 = y2 * width + x2
        pixels[pos1], pixels[pos2] = pixels[pos2], pixels[pos1]
    new_image.putdata(pixels)
    new_image.save(pic_name + '.jpg')


# 按像素随机混合，越左边取到left_image的像素越多，越右边取到right_image的像素越多
def pixelMix(left_image, right_image, pic_name):
    width, height = left_image.size
    new_image = Image.new('RGB', (width, height))
    leftPixels = list(left_image.getdata())
    rightPixels = list(right_image.getdata())
    colInsertIndexs = []
    for i in range(width):
        colInsertIndexPos = random.sample(range(height), height * i // width)
        colInsertIndexs.append(colInsertIndexPos)

    pixels = []
    for i in range(width):
        for j in range(height):
            colInsertIndexPos = colInsertIndexs[i]
            if j in colInsertIndexPos:
                pixels.append(rightPixels[i * width + j])
            else:
                pixels.append(leftPixels[i * width + j])
    new_image.putdata(pixels)
    new_image.save(pic_name + '.jpg')


# 旋转图片
def picRotated(base_image, pic_name, angle):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    new_image.paste(base_image.rotate(angle), (0, 0))
    new_image.save(pic_name + '.jpg')


# 漩涡式旋转图片，离中心越远旋转越多
def picRotatedEddies(base_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    center_x, center_y = width // 2, height // 2
    maxDistance = math.sqrt(center_x ** 2 + center_y ** 2) / 2

    for x in range(width):
        for y in range(height):
            # 计算像素点与中心点的距离
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # 计算旋转角度, 最大为180度
            angle = (maxDistance - distance) / maxDistance * 180 if distance <= maxDistance else 0
            
            # 对每个像素点进行旋转变换
            new_x = int(center_x + (x - center_x) * math.cos(math.radians(angle)) - (y - center_y) * math.sin(math.radians(angle)))
            new_y = int(center_y + (x - center_x) * math.sin(math.radians(angle)) + (y - center_y) * math.cos(math.radians(angle)))
            new_x = min(max(0, new_x), width - 1)
            new_y = min(max(0, new_y), height - 1)
            
            # 获取旋转后的像素值
            pixel = image.getpixel((new_x, new_y))
            new_image.putpixel((x, y), pixel)

    new_image.save(pic_name + '.jpg')


# 按块复制(中心图格复制到周围一圈)
def blockRepeat(base_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    block_width = width // 54
    block_height = height // 54
    blocks = []
    for i in range(54):
        lineBlocks = []
        for j in range(54):
            box = (i * block_width, j * block_height, (i + 1) * block_width, (j + 1) * block_height)
            lineBlocks.append(base_image.crop(box))
        blocks.append(lineBlocks)

    for i in range(54):
        for j in range(54):
            useBlockINum = i + 1 - i % 3
            useBlockJNum = j + 1 - j % 3
            new_image.paste(blocks[useBlockINum][useBlockJNum], (i * block_width, j * block_height))

    new_image.save(pic_name + '.jpg')


# 按块随机打乱
def blockRandomChange(base_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    blockNumX, blockNumY = 12, 12
    block_width = width // blockNumX
    block_height = height // blockNumY
    blocks = []
    for i in range(blockNumX):
        lineBlocks = []
        for j in range(blockNumY):
            box = (i * block_width, j * block_height, (i + 1) * block_width, (j + 1) * block_height)
            lineBlocks.append(base_image.crop(box))
        blocks.append(lineBlocks)

    blockPosIndex = random.sample(range(blockNumX * blockNumY), blockNumX * blockNumY)
    for i in range(blockNumX):
        for j in range(blockNumY):
            new_image.paste(blocks[blockPosIndex[i * blockNumX + j] // blockNumX][blockPosIndex[i * blockNumX + j] % blockNumX], (i * block_width, j * block_height))

    new_image.save(pic_name + '.jpg')


# 按块打乱，从左到右打乱程度逐渐增加
def progressively_randomize(base_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    new_image.paste(base_image, (0, 0))
    blockNumX, blockNumY = 12, 12
    block_width = width // blockNumX
    block_height = height // blockNumY
    
    for i in range(blockNumX):
        newIndexs = list(range(blockNumY))
        for _ in range(i // 2):
            swapIndexs = random.sample(range(blockNumY), 2)
            newIndexs[swapIndexs[0]], newIndexs[swapIndexs[1]] = newIndexs[swapIndexs[1]], newIndexs[swapIndexs[0]]
        # x轴按照newIndexs的顺序排列
        for oldIndex in range(blockNumY):
            new_index = newIndexs[oldIndex]
            box = (i * block_width, oldIndex * block_height, (i + 1) * block_width, (oldIndex + 1) * block_height)
            new_image.paste(base_image.crop(box), (i * block_width, new_index * block_height))
    
    new_image.save(pic_name + '.jpg')


# B图片按像素随机插入A图片
def pixelRandomInsert(base_image, insert_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    pixels = list(base_image.getdata())
    insert_pixels = list(insert_image.getdata())
    insert_pos = random.sample(range(width * height), width * height // 4)
    for pos in insert_pos:
        pixels[pos] = insert_pixels[pos]
    new_image.putdata(pixels)
    new_image.save(pic_name + '.jpg')


# B图片按块随机插入A图片
def blockRandomInsert(base_image, insert_image, pic_name):
    width, height = base_image.size
    new_image = Image.new('RGB', (width, height))
    blockNumX, blockNumY = 100, 100
    block_width = width // blockNumX
    block_height = height // blockNumY
    blocks = []
    blocksInsert = []
    for i in range(blockNumX):
        lineBlocks = []
        lineBlocksInsert = []
        for j in range(blockNumY):
            box = (i * block_width, j * block_height, (i + 1) * block_width, (j + 1) * block_height)
            lineBlocks.append(base_image.crop(box))
            boxInsert = (i * block_width, j * block_height, (i + 1) * block_width, (j + 1) * block_height)
            lineBlocksInsert.append(insert_image.crop(boxInsert))
        blocks.append(lineBlocks)
        blocksInsert.append(lineBlocksInsert)

    insertPosIndex = random.sample(range(width * height), width * height // 2)
    for i in range(blockNumX):
        for j in range(blockNumY):
            if i * blockNumX + j in insertPosIndex:
                new_image.paste(blocksInsert[i][j], (i * block_width, j * block_height))
            else:
                new_image.paste(blocks[i][j], (i * block_width, j * block_height))

    new_image.save(pic_name + '.jpg')


# 保存三通道图片（保留本通道颜色，其他通道置0）
def decompose_rgb_image(base_array, pic_name):
    width, height = image.size
    image_r_display = Image.new('RGB', (width, height))
    image_r_display.putdata([(x, 0, 0) for x in base_array[:, :, 0].flatten()])
    image_r_display.save(pic_name + '_r.jpg')
    image_g_display = Image.new('RGB', (width, height))
    image_g_display.putdata([(0, x, 0) for x in base_array[:, :, 1].flatten()])
    image_g_display.save(pic_name + '_g.jpg')
    image_b_display = Image.new('RGB', (width, height))
    image_b_display.putdata([(0, 0, x) for x in base_array[:, :, 2].flatten()])
    image_b_display.save(pic_name + '_b.jpg')


# r通道每个像素中心对称，g、b通道不变，然后叠加
def save_r_filter_image(base_array1, base_array2, pic_name):
    image_r_array = base_array2[:, :, 0]
    image_r_array = np.fliplr(image_r_array)
    image_r_array = np.flipud(image_r_array)
    image_r_display = Image.fromarray(image_r_array)
    image_g_array = base_array1[:, :, 1]
    image_g_display = Image.fromarray(image_g_array)
    image_b_array = base_array1[:, :, 2]
    image_b_display = Image.fromarray(image_b_array)
    new_image = Image.merge('RGB', (image_r_display, image_g_display, image_b_display))
    new_image.save(pic_name + '.jpg')


# 对RGB三通道图片，进行FFT低通滤波
def fft_filter_image_rgb(base_array, pic_name):
    single_channel_images = []
    for i in range(3):
        single_channel_array = base_array[:, :, i]
        fft_array = np.fft.fft2(single_channel_array)
        fft_array = np.fft.fftshift(fft_array)
        # 添加一个低通滤波
        for i in range(fft_array.shape[0]):
            for j in range(fft_array.shape[1]):
                distance = (i - fft_array.shape[0] // 2) ** 2 + (j - fft_array.shape[1] // 2) ** 2
                if distance < 20000:
                    fft_array[i][j] /= 5
        fft_array = np.fft.ifftshift(fft_array)
        single_channel_image = Image.fromarray(np.uint8(np.abs(np.fft.ifft2(fft_array))))
        single_channel_images.append(single_channel_image)
    new_image = Image.merge('RGB', (single_channel_images[0], single_channel_images[1], single_channel_images[2]))
    new_image.save(pic_name + '.jpg')


# 对灰度图进行FFT低通滤波
def fft_filter_image_gray(base_array, pic_name):
    fft_array = np.fft.fft2(base_array)
    fft_array = np.fft.fftshift(fft_array)
    # 添加一个低通滤波
    for i in range(fft_array.shape[0]):
        for j in range(fft_array.shape[1]):
            distance = (i - fft_array.shape[0] // 2) ** 2 + (j - fft_array.shape[1] // 2) ** 2
            if distance < 20000:
                fft_array[i][j] /= 5
    fft_array = np.fft.ifftshift(fft_array)
    new_image = Image.fromarray(np.uint8(np.abs(np.fft.ifft2(fft_array))))
    new_image.save(pic_name + '.jpg')


# 灰度图边缘提取(Laplacian 5*5)
def laplacian_edge_detection(base_array, pic_name):
    filter_array = np.array([
        [0, 1, 0, 1, 0],
        [1, -4, 1, -4, 1],
        [0, 1, 0, 1, 0],
        [1, -4, 1, -4, 1],
        [0, 1, 0, 1, 0]
    ])
    new_array = np.zeros(base_array.shape)
    for i in range(2, base_array.shape[0] - 2):
        for j in range(2, base_array.shape[1] - 2):
            new_array[i][j] = np.sum(base_array[i - 2:i + 3, j - 2:j + 3] * filter_array)
    new_image = Image.fromarray(np.uint8(new_array))
    new_image.save(pic_name + '.jpg')


# 灰度图边缘提取(Canny 5*5)
def canny_edge_detection(base_array, pic_name):
    filter_array = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  1,  2,  1, -1],
        [-1,  2,  4,  2, -1],
        [-1,  1,  2,  1, -1],
        [-1, -1, -1, -1, -1]
    ])
    new_array = np.zeros(base_array.shape)
    for i in range(2, base_array.shape[0] - 2):
        for j in range(2, base_array.shape[1] - 2):
            new_array[i][j] = np.sum(base_array[i - 2:i + 3, j - 2:j + 3] * filter_array)
            if new_array[i][j] > 0:
                new_array[i][j] = 255
    new_image = Image.fromarray(np.uint8(new_array))
    new_image.save(pic_name + '.jpg')


if __name__ == '__main__':
    image = Image.open('image/you1.jpg')
    # image_array = np.array(image)
    picRotated(image, 'you1_rotated', -45)
