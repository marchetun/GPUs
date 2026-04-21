CXX=icpx
CXXFLAGS= -fsycl -fsycl-targets=nvptx64-nvidia-cuda "-xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89"
LIBS=

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o