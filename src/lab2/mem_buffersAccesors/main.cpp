#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

int main(int argc, char **argv) {

    if (argc != 2) {
        std::cout << "Uso: ./exec N" << std::endl;
        return -1;
    }

    int N = std::stoi(argv[1]);

    // sycl::gpu_selector_v es la forma correcta en SYCL 2020
    queue Q(cpu_selector_v);

    std::cout << "Running on: "
              << Q.get_device().get_info<info::device::name>()
              << std::endl;

    // Usamos un std::vector para gestionar la memoria en el Host de forma segura
    std::vector<float> a(N);
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i); // Inicialización
    }

        buffer<float, 1> buffer_a(a.data(), range<1>(N));

        Q.submit([&](handler &h) {
            // Los accessors ahora son más concisos
            accessor acc_a(buffer_a, h, read_write);

            h.parallel_for(range<1>(N), [=](id<1> i) {
                acc_a[i] *= 2.0f;
            });
        }); 

    // Imprimir resultados
    for (int i = 0; i < N; i++) {
        std::cout << "a[" << i << "] = " << a[i] << std::endl;
    }

    return 0;
}
