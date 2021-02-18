
#include <CL/sycl.hpp>
#include <stdio.h>

using namespace cl::sycl;

int main(int argc, char* argv[]) {
    const int FILAS = 9;
    const int COLS = 9;

    clock_t start, end;

    try {
        // cola de trabajo
        sycl::queue q(sycl::default_selector{});//GPU(si hay) o CPU
        //sycl::queue q(sycl::cpu_selector{});
        //sycl::queue q(sycl::gpu_selector{});

        // presenta información del dispositivo usado
        std::cout << "Dispositivo: " << q.get_device().get_info<info::device::name>() << std::endl;

        //USM para guardar los datos
        int* matriz = malloc_shared<int>(FILAS * COLS, q);//matriz (2D) vectorizada (1D)

        // inicia cronómetro
        start = clock();

        q.parallel_for(sycl::range<2>{FILAS, COLS}, [=](sycl::id<2> i) {
            int row = i[0];
            int col = i[1];
            matriz[row * FILAS + col] = row * 10 + col;
        }).wait();

        // finaliza cronómetro
        end = clock();

        // presenta el resultado
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%2d, ", matriz[i * FILAS + j]);//recuerden que la matriz está vectorizada!
            }
            printf("\n");
        }
        // tiempo total de ejecución de la kernel
        double time_taken = double(end - start);
        printf("Segundos: %f\n", time_taken / CLOCKS_PER_SEC);
    }
    catch (sycl::exception& e) {
        printf("Problemas !!!: %s\n", e.what());
        return 1;
    }

    return 0;
}
