OBJDIR=obj
OUTPUTDIR=out
TESTDIR=test

LIBDIR=-L$(OUTPUTDIR) -L/u/yujiali/Desktop/Develop/cuda/lib
LIBS=-lcudamatconv -lcgpulock

all: $(OUTPUTDIR)/libcudamatconv.so $(OUTPUTDIR)/test

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OUTPUTDIR):
	mkdir -p $(OUTPUTDIR)

$(OUTPUTDIR)/libcudamatconv.so: cudamat_conv.cu cudamat_conv.cuh cudamat.cu cudamat.cuh cudamat_kernels.cu cudamat_kernels.cuh cudamat_conv_kernels.cu cudamat_conv_kernels.cuh | $(OUTPUTDIR)
	nvcc -O --compiler-options '-fPIC' -o $(OUTPUTDIR)/libcudamatconv.so --shared cudamat_conv.cu cudamat_conv_kernels.cu cudamat.cu cudamat_kernels.cu -lcublas
	#nvcc -O --ptxas-options=-v --compiler-options '-fPIC' -o $(OUTPUTDIR)/libcudamatconv.so --shared cudamat_conv.cu cudamat_conv_kernels.cu cudamat.cu cudamat_kernels.cu -lcublas

$(OBJDIR)/test_helper.o: $(TESTDIR)/test_helper.cpp $(TESTDIR)/test_helper.h | $(OBJDIR)
	g++ -c $(TESTDIR)/test_helper.cpp -o $(OBJDIR)/test_helper.o

$(OUTPUTDIR)/test: $(TESTDIR)/test.cpp $(OBJDIR)/test_helper.o $(OUTPUTDIR)/libcudamatconv.so | $(OUTPUTDIR)
	nvcc $(LIBDIR) $(OBJDIR)/test_helper.o $(TESTDIR)/test.cpp -o $(OUTPUTDIR)/test $(LIBS)

clean:
	rm -f $(OBJDIR)/*

rm:
	rm -f $(OBJDIR)/*
	rm -f $(OUTPUTDIR)/*

