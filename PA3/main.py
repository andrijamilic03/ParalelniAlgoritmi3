import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os


mod = SourceModule("""
__global__ void grayscale(unsigned char *img_in, unsigned char *img_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = img_in[idx];
        unsigned char g = img_in[idx + 1];
        unsigned char b = img_in[idx + 2];

        unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

        img_out[y * width + x] = gray;
    }
}

__global__ void gaussian_blur(unsigned char *img_in, unsigned char *img_out, int width, int height, float *kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = threadIdx.z;  

    // Deljena memorija za skladistenje kernela 
    __shared__ float shared_kernel[1024];
    int shared_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Ucitaj kernel u deljenu memoriju
    if (shared_idx < kernel_size * kernel_size) {
        shared_kernel[shared_idx] = kernel[shared_idx];
    }
    __syncthreads();

    if (x < width && y < height) {
        float new_val = 0.0f;

        for (int ky = -kernel_size / 2; ky <= kernel_size / 2; ++ky) {
            for (int kx = -kernel_size / 2; kx <= kernel_size / 2; ++kx) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                int idx = (ny * width + nx) * 3 + z;

                // Koristi deljeni kernel
                float kernel_val = shared_kernel[(ky + kernel_size / 2) * kernel_size + (kx + kernel_size / 2)];

                new_val += img_in[idx] * kernel_val;
            }
        }
        int idx_out = (y * width + x) * 3 + z;
        img_out[idx_out] = min(max(int(new_val), 0), 255);
    }
}

__global__ void calculate_average(unsigned char *img_in, int width, int height, float *average) {
    extern __shared__ float block_sum[]; // Shared memorija za sumu unutar bloka
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x; // Indeks niti u bloku

    if (thread_idx == 0) block_sum[0] = 0.0f;
    __syncthreads();

    float pixel_sum = 0.0f;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = img_in[idx];
        unsigned char g = img_in[idx + 1];
        unsigned char b = img_in[idx + 2];

        pixel_sum = r + g + b;
    }

    atomicAdd(&block_sum[0], pixel_sum);
    __syncthreads();

    if (thread_idx == 0) {
        atomicAdd(average, block_sum[0]);
    }

}


__global__ void adjust_brightness(unsigned char *img_in, unsigned char *img_out, int width, int height, float factor, float average) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r = img_in[idx];
        float g = img_in[idx + 1];
        float b = img_in[idx + 2];

        r = average + (r - average) * factor;
        g = average + (g - average) * factor;
        b = average + (b - average) * factor;

        img_out[idx] = min(max(int(r), 0), 255);
        img_out[idx + 1] = min(max(int(g), 0), 255);
        img_out[idx + 2] = min(max(int(b), 0), 255);
    }
}

""")

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def gaussian_blur(img, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma).astype(np.float32)
    width, height = img.shape[1], img.shape[0]
    img_in = np.asarray(img, dtype=np.uint8)
    img_out = np.zeros_like(img, dtype=np.uint8)

    img_in_gpu = cuda.mem_alloc(img_in.nbytes)
    img_out_gpu = cuda.mem_alloc(img_out.nbytes)
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)

    cuda.memcpy_htod(img_in_gpu, img_in)
    cuda.memcpy_htod(kernel_gpu, kernel)

    gaussian_kernel_func = mod.get_function("gaussian_blur")

    block_size = (16, 16, 3)
    grid_size = (int(np.ceil(width / 16)), int(np.ceil(height / 16)))

    gaussian_kernel_func(img_in_gpu, img_out_gpu, np.int32(width), np.int32(height), kernel_gpu, np.int32(kernel_size),
                         block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(img_out, img_out_gpu)

    return img_out


def grayscale(img):
    width, height = img.shape[1], img.shape[0]
    img_in = np.asarray(img, dtype=np.uint8)
    img_out = np.zeros((height, width), dtype=np.uint8)

    img_in_gpu = cuda.mem_alloc(img_in.nbytes)
    img_out_gpu = cuda.mem_alloc(img_out.nbytes)

    cuda.memcpy_htod(img_in_gpu, img_in)

    grayscale_kernel = mod.get_function("grayscale")

    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(width / 16)), int(np.ceil(height / 16)))

    grayscale_kernel(img_in_gpu, img_out_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(img_out, img_out_gpu)

    return img_out


def adjust_brightness(img, factor):
    width, height = img.shape[1], img.shape[0]
    img_in = np.asarray(img, dtype=np.uint8)
    img_out = np.zeros_like(img, dtype=np.uint8)

    img_in_gpu = cuda.mem_alloc(img_in.nbytes)
    img_out_gpu = cuda.mem_alloc(img_out.nbytes)
    average_gpu = cuda.mem_alloc(np.float32().nbytes)

    cuda.memcpy_htod(img_in_gpu, img_in)

    calculate_avg_kernel = mod.get_function("calculate_average")
    adjust_brightness_kernel = mod.get_function("adjust_brightness")

    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(width / 16)), int(np.ceil(height / 16)))
    average = np.zeros(1, dtype=np.float32)

    calculate_avg_kernel(img_in_gpu, np.int32(width), np.int32(height), average_gpu, block=block_size, grid=grid_size,
                         shared=block_size[0] * block_size[1] * 4)

    cuda.memcpy_dtoh(average, average_gpu)

    average[0] /= (width * height * 3)

    adjust_brightness_kernel(
        img_in_gpu, img_out_gpu, np.int32(width), np.int32(height), np.float32(factor), np.float32(average[0]),
        block=block_size, grid=grid_size
    )

    cuda.memcpy_dtoh(img_out, img_out_gpu)

    return img_out

def main():
    if not os.path.exists('test'):
        os.makedirs('test')

    img = cv2.imread('slike/slika.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred_img = gaussian_blur(img, kernel_size=5, sigma=2)
    cv2.imwrite('test/blurred_image.jpg', cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR))

    grayscale_img = grayscale(img)
    cv2.imwrite('test/grayscale_image.jpg', grayscale_img)

    brightness_img = adjust_brightness(img, factor=2)
    cv2.imwrite('test/brightness_image.jpg', cv2.cvtColor(brightness_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()