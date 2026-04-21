
#include <sycl/sycl.hpp>

using  namespace  sycl;


void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
	Q.submit([&](handler &h){

		h.parallel_for(range<2>(height, width), [=](id<2> item){
    		int i = item[0]; // Fila
    		int j = item[1]; // Columna

    		int ws2 = (window_size - 1) >> 1; // Sacamos el radio, desplazamiento a la derecha en vez de divisón por eficiencia

    // Control de seguridad para no salirse de la imagen
    if (i >= ws2 && i < height - ws2 && j >= ws2 && j < width - ws2) {
		const int MAX_WINDOW = 25; // Suficiente para 5x5
		float window[MAX_WINDOW];
		int n = window_size * window_size;
		int counter = 0;

		// Llenar ventana
		for(int ii = i - ws2; ii <= i + ws2; ii++ ){
			for(int jj = j - ws2; jj <= j + ws2; jj++ ){
				window[counter++] = im[ii * width + jj];
			}
		}

		// Bubble Sort
		for (int a = 0; a < n - 1; a++) {
			for (int b = 0; b < n - a - 1; b++) {
				if (window[b] > window[b + 1]) {
					float temp = window[b];
					window[b] = window[b + 1];
					window[b + 1] = temp;
				}
			}
		}

		// Lógica de mediana y threshold
		float median = window[n >> 1]; // El elemento central
		float pixel_original = im[i * width + j];

		// Usamos sycl::fabs para el valor absoluto en el device
		if (sycl::fabs((median - pixel_original) / median) <= thredshold) {
			image_out[i * width + j] = pixel_original;
		} else {
			image_out[i * width + j] = median;
		}
    } else {
        // Opcional: copiar píxel original en los bordes
        image_out[i * width + j] = im[i * width + j];
    }
});
}).wait();
}
