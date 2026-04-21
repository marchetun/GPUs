/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GSimulation.hpp"
#include "cpu_time.hpp"

GSimulation :: GSimulation(sycl::queue Q)
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000); 
  set_nsteps(10);
  set_tstep(0.1); 
  set_sfreq(1);
  _Q = Q;//añadimos la cola
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()  
{
  std::random_device rd;	//random number generator
  std::mt19937 gen(42);      
  std::uniform_real_distribution<real_type> unif_d(0,1.0);
  
  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
  }
}

void GSimulation :: init_vel()  
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[2] = unif_d(gen) * 1.0e-3f; 
  }
}

void GSimulation :: init_acc() 
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].acc[0] = 0.f; 
    particles[i].acc[1] = 0.f;
    particles[i].acc[2] = 0.f;
  }
}

void GSimulation :: init_mass() 
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].mass = n * unif_d(gen); 
  }
}

void GSimulation::get_acceleration(int n)
{
    // Extraemos el puntero como en el updateParticles
    auto *particles_ptr = this->particles;
    const float softeningSquared = 1e-3f;
    const float G = 6.67259e-11f;

    _Q.submit([&](sycl::handler &h) {
        // Tamaño del bloque
        const int B = 256; 
        
        // Declaramos el Local Accessor
        sycl::local_accessor<real_type, 1> cache_pos_x(sycl::range<1>(B), h);
        sycl::local_accessor<real_type, 1> cache_pos_y(sycl::range<1>(B), h);
        sycl::local_accessor<real_type, 1> cache_pos_z(sycl::range<1>(B), h);
        sycl::local_accessor<real_type, 1> cache_mass(sycl::range<1>(B), h);

        // Definimos el nd_range: global y local
        sycl::nd_range<1> range{sycl::range<1>(n), sycl::range<1>(B)};

        h.parallel_for(range, [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0);
            int local_id = item.get_local_id(0);

            // Variables en registros para acumular la aceleración
            real_type ax_i = 0.0f;
            real_type ay_i = 0.0f;
            real_type az_i = 0.0f;

            // Guardamos la posición de la partícula i en registros locales
            real_type pos_i_x = particles_ptr[i].pos[0];
            real_type pos_i_y = particles_ptr[i].pos[1];
            real_type pos_i_z = particles_ptr[i].pos[2];

            // Bucle externo por Bloques 
            for (int j_block = 0; j_block < n; j_block += B) {
                
                // Carga cooperativa al cache local
                int j_global = j_block + local_id;
                cache_pos_x[local_id] = particles_ptr[j_global].pos[0];
                cache_pos_y[local_id] = particles_ptr[j_global].pos[1];
                cache_pos_z[local_id] = particles_ptr[j_global].pos[2];
                cache_mass[local_id]  = particles_ptr[j_global].mass;

                // Esperamos a que todos hayan cargado
                item.barrier(sycl::access::fence_space::local_space);

                // Cómputo usando la memoria local
                for (int j = 0; j < B; j++) {
                    real_type dx = cache_pos_x[j] - pos_i_x;
                    real_type dy = cache_pos_y[j] - pos_i_y;
                    real_type dz = cache_pos_z[j] - pos_i_z;

                    real_type distSqr = dx*dx + dy*dy + dz*dz + softeningSquared;
                    real_type invDist = 1.0f / sycl::sqrt(distSqr);
                    real_type invDist3 = invDist * invDist * invDist;

                    real_type s = cache_mass[j] * G * invDist3;
                    ax_i += dx * s;
                    ay_i += dy * s;
                    az_i += dz * s;
                }

                // Esperamos antes de cargar el siguiente bloque
                item.barrier(sycl::access::fence_space::local_space);
            }

            // Escritura única a memoria global
            particles_ptr[i].acc[0] = ax_i;
            particles_ptr[i].acc[1] = ay_i;
            particles_ptr[i].acc[2] = az_i;
        });
    }).wait();
}
real_type GSimulation::updateParticles(int n, real_type dt)
{
    // Reserva memoria shared, quizas podría funcionar con memoria local si ejecuto en cpu
    real_type *energy_ptr = sycl::malloc_shared<real_type>(1, _Q);
    *energy_ptr = 0.0f; // Inicialización con literal float para evitar conflictos de tipos

    //Extraemos el puntero a una variable local, no pudo acceder a this desde el kernel
    auto *particles_ptr = this->particles; // ya tiene malloc shared

    _Q.submit([&](sycl::handler &h) {
        // Definimos el objeto de reducción. 
        // Usamos 0.0f para coincidir con el tipo real_type.
        auto red = sycl::reduction(energy_ptr, 0.0f, sycl::plus<>());

        h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> id, auto &reducer) {
            int i = id[0];

            // Cómputo de los hilos
            particles_ptr[i].vel[0] += particles_ptr[i].acc[0] * dt;
            particles_ptr[i].vel[1] += particles_ptr[i].acc[1] * dt;
            particles_ptr[i].vel[2] += particles_ptr[i].acc[2] * dt;

            particles_ptr[i].pos[0] += particles_ptr[i].vel[0] * dt;
            particles_ptr[i].pos[1] += particles_ptr[i].vel[1] * dt;
            particles_ptr[i].pos[2] += particles_ptr[i].vel[2] * dt;

            particles_ptr[i].acc[0] = 0.0f;
            particles_ptr[i].acc[1] = 0.0f;
            particles_ptr[i].acc[2] = 0.0f;

            // Cálculo de energía para la reducción
            real_type energy_thread = particles_ptr[i].mass * (
                particles_ptr[i].vel[0] * particles_ptr[i].vel[0] +
                particles_ptr[i].vel[1] * particles_ptr[i].vel[1] +
                particles_ptr[i].vel[2] * particles_ptr[i].vel[2]);

            // Aplicamo reducción usando el reducer
            reducer.combine(energy_thread);
        });
    }).wait(); // Esperamos a que la GPU termine antes de leer el resultado.

    //Recuperamos el valor y liberamos la memoria temporal
    real_type total_energy = *energy_ptr;
    sycl::free(energy_ptr, _Q);

    return total_energy;
}
void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  //allocate particles, ahora reservamos con memoria compartida, aunque si ejecutamos en cpu podría funcionar
  particles = sycl::malloc_shared<ParticleAoS>(n, _Q);

  init_pos();
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
  
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   ts0 += time.start(); 


    get_acceleration(n);

    energy = updateParticles(n, dt);
    _kenergy = 0.5 * energy; 
    
    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2) 
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } //end of the time step loop
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  

  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  sycl::free(particles, _Q);
}
