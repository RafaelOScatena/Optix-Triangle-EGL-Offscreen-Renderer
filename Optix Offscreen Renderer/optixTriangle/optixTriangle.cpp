//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// Some helper macros to stringify the sample's name that comes in as a define
#define OPTIX_STRINGIFY2(name) #name
#define OPTIX_STRINGIFY(name) OPTIX_STRINGIFY2(name)
#define OPTIX_SAMPLE_NAME OPTIX_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)
#define OPTIX_SAMPLE_DIR OPTIX_STRINGIFY(OPTIX_SAMPLE_DIR_DEFINE)


#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <CUDAOutputBuffer.h>
//#include <sutil/Exception.h>
//#include <sutil/sutil.h>

#include "optixTriangle.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

//#include <sutil/Camera.h>
//#include <sutil/Trackball.h>

//#include <EGL/egl.h>
//#include <EGL/eglext.h>
//#include <GLES/gl.h>


//#include <GL/gl.h>
//#include <GL/glext.h>

//#include <sutilapi.h>
#include <stdio.h>
//#include <GLES2/gl2.h>
#include <png.h>
#include <stdlib.h>
#include <map>

#include <fstream>
#include <vector>
#include <iterator>
#include <Camera.h>

#include <stdexcept>
#define GL_GLEXT_PROTOTYPES 1





void parseDimensions( const char* arg, int& width, int& height )
{
    // look for an 'x': <width>x<height>
    size_t width_end    = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char* height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        width  = atoi( width_arg );
        height = atoi( height_arg );
        return;
    }
    const std::string err = "Failed to parse width, height from string '" + std::string( arg ) + "'";
    throw std::invalid_argument( err.c_str() );
}

static bool readSourceFile( std::string& str, const std::string& filename )
{
    // Try to open file
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}


// Returns string of file extension including '.'
static std::string fileExtensionForLoading()
{
    std::string extension;
#if SAMPLES_INPUT_GENERATE_PTX
    extension = ".ptx";
#endif
#if SAMPLES_INPUT_GENERATE_OPTIXIR
    extension = ".optixir";
#endif
    if( const char* ext = getenv("OPTIX_SAMPLES_INPUT_EXTENSION") )
    {
        extension = ext;
        if( extension.size() && extension[0] != '.' )
            extension = "." + extension;
    }
    return extension;
}



static std::string sampleInputFilePath( const char* sampleName, const char* fileName )
{
    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_PTX_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_PTX_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_PTX_DIR" ),
 #if defined(CMAKE_INTDIR)
        SAMPLES_PTX_DIR "/" CMAKE_INTDIR,
#endif
        SAMPLES_PTX_DIR,
        "."
    };

    // Allow overriding the file extension
    std::string extension = fileExtensionForLoading();

    if( !sampleName )
        sampleName = "sutil";
    std::vector<std::string> locations;
    for( const char* directory : directories )
    {
        if( directory )
        {
            std::string path = directory;
            path += '/';
            path += sampleName;
            path += "_generated_";
            path += fileName;
            path += extension;
            locations.push_back( path );
            if( fileExists( path ) )
                return path;
        }
    }

    std::string error = "sutil::sampleInputFilePath couldn't locate " + std::string( fileName ) + " for sample "
                        + std::string( sampleName ) + " in the following locations:\n";
    for( const auto& path : locations )
        error += "\t" + path + "\n";

    throw eglrender::Exception( error.c_str() );
}


static void getInputDataFromFile( std::string& inputData, const char* sample_name, const char* filename )
{
    const std::string sourceFilePath = sampleInputFilePath( sample_name, filename );

    // Try to open source file
    if( !readSourceFile( inputData, sourceFilePath ) )
    {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error( err.c_str() );
    }
}



struct SourceCache
{
    std::map<std::string, std::string*> map;
    ~SourceCache()
    {for( std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it )
            delete it->second;}};
         
           

static SourceCache g_sourceCache;

const char* getInputData( const char*                     sample,
                          const char*                     sampleDir,
                          const char*                     filename,
                          size_t&                         dataSize,
                          const char**                    log,
                          const std::vector<const char*>& compilerOptions )
{
    if( log )
        *log = NULL;

    std::string *                                 inputData, cu;
    std::string                                   key  = std::string( filename ) + ";" + ( sample ? sample : "" );
    std::map<std::string, std::string*>::iterator elem = g_sourceCache.map.find( key );

    if( elem == g_sourceCache.map.end() )
    {
        inputData = new std::string();
#if CUDA_NVRTC_ENABLED
        std::string location;
        getCuStringFromFile( cu, location, sampleDir, filename );
        getInputFromCuString( *inputData, sampleDir, cu.c_str(), location.c_str(), log, compilerOptions );
#else
        getInputDataFromFile( *inputData, sample, filename );
#endif
        g_sourceCache.map[key] = inputData;
    }
    else
    {
        inputData = elem->second;
    }
    dataSize = inputData->size();
    return inputData->c_str();
}


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


void configureCamera( eglrender::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, 2.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}


int main( int argc, char* argv[] )
{
    std::string outfile;
    int         width  = 1024;
    int         height =  768;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
#ifdef DEBUG
            // This may incur significant performance cost and should only be done during development.
            options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }


        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // Triangle build input: simple list of three vertices
            const std::array<float3, 3> vertices =
            { {
                  { -0.5f, -0.5f, 0.0f },
                  {  0.5f, -0.5f, 0.0f },
                  {  0.0f,  0.5f, 0.0f }
            } };

            const size_t vertices_size = sizeof( float3 )*vertices.size();
            CUdeviceptr d_vertices=0;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_vertices ),
                        vertices.data(),
                        vertices_size,
                        cudaMemcpyHostToDevice
                        ) );

            // Our build input is a simple list of non-indexed triangle vertices
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
            triangle_input.triangleArray.vertexBuffers = &d_vertices;
            triangle_input.triangleArray.flags         = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage(
                        context,
                        &accel_options,
                        &triangle_input,
                        1, // Number of build inputs
                        &gas_buffer_sizes
                        ) );
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_temp_buffer_gas ),
                        gas_buffer_sizes.tempSizeInBytes
                        ) );
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_gas_output_buffer ),
                        gas_buffer_sizes.outputSizeInBytes
                        ) );

            OPTIX_CHECK( optixAccelBuild(
                        context,
                        0,                  // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,                  // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_gas_output_buffer,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        nullptr,            // emitted property list
                        0                   // num emitted properties
                        ) );

            // We can now free the scratch space buffer used during build and the vertex
            // inputs, since they are not needed by our trivial shading method
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
        }

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
            module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 3;
            pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t      inputSize  = 0;
            const char* input      = getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixTriangle.cu", inputSize, NULL, {CUDA_NVRTC_OPTIONS});

            OPTIX_CHECK_LOG( optixModuleCreate(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        input,
                        inputSize,
                        LOG, &LOG_SIZE,
                        &module
                        ) );
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &hitgroup_prog_group
                        ) );
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = max_trace_depth;
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        LOG, &LOG_SIZE,
                        &pipeline
                        ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
            {
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                     0,  // maxCCDepth
                                                     0,  // maxDCDEpth
                                                     &direct_callable_stack_size_from_traversal,
                                                     &direct_callable_stack_size_from_state, &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( hitgroup_record ),
                        &hg_sbt,
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = 1;
        }

        eglrender::CUDAOutputBuffer<uchar4> output_buffer( eglrender::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
        
    std::cerr << "before launch" << std::endl;

   /*     

	    EGLDisplay display;
	    EGLConfig config;
	    EGLContext context;
	    EGLSurface surface;
	    EGLint num_config;        
        
    
        // 1. Initialize EGL display connection
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY) {printf("Failed to get EGL display.\n");return -1;}
        if (!eglInitialize(display, NULL, NULL)) {printf("Failed to initialize EGL.\n");
            return -1;}



        // 2. Choose EGL config
        EGLint configAttributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RED_SIZE, 8,            EGL_GREEN_SIZE, 8,            EGL_BLUE_SIZE, 8,
            EGL_DEPTH_SIZE, 24,            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_NONE        };

        EGLint numConfigs;
        if (!eglChooseConfig(display, configAttributes, &config, 1, &numConfigs) || numConfigs == 0) {
            printf("Failed to choose EGL config.\n");            return -1;
        }

        // 3. Create an EGL context
        EGLint contextAttributes[] = {
            EGL_CONTEXT_CLIENT_VERSION, 2, // For OpenGL ES 2.0
            EGL_NONE        };

        context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttributes);
        if (context == EGL_NO_CONTEXT) {
       
        printf("Failed to create EGL context.\n");
        return -1;
        }
        
        eglChooseConfig(display, nullptr, &config, 1, &num_config);
	    assertEGLError("eglChooseConfig");
	    
	    eglBindAPI(EGL_OPENGL_API);
	    assertEGLError("eglBindAPI");
	    
	    context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
	    assertEGLError("eglCreateContext");

	    //surface = eglCreatePbufferSurface(display, config, nullptr);
	    //assertEGLError("eglCreatePbufferSurface");
	    
	    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
	    assertEGLError("eglMakeCurrent");



        // 4. Create an EGL pbuffer surface (offscreen)
        EGLint surfaceAttributes[] = {EGL_WIDTH, 1,EGL_HEIGHT, 1, EGL_NONE };

//        EGLSurface surface = eglCreatePbufferSurface(display, config, surfaceAttributes);
//        if (surface == EGL_NO_SURFACE) {
//            printf("Failed to create EGL pbuffer surface.\n");return -1;        }

        // 5. Make the EGL context and surface current
        if (!eglMakeCurrent(display, surface, surface, context)) {
            printf("Failed to make EGL context and surface current.\n");
            return -1;}*/


        // Now the EGL context is current and ready for rendering.

       
            CUstream stream;// Declare eglStream if it's not already declared
 
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            eglrender::Camera cam;
            configureCamera( cam, width, height );

            Params params;
            params.image        = output_buffer.map();
            params.image_width  = width;
            params.image_height = height;
            params.handle       = gas_handle;
            params.cam_eye      = cam.eye();
            cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );

            std::cerr << "params Ok" << std::endl;

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_param ),&params, sizeof( params ),
                        cudaMemcpyHostToDevice) );

        // Launch Optix kernel
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));

        // Synchronize CUDA streams
        CUDA_SYNC_CHECK();

        output_buffer.unmap();
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        
        std::cerr << "Cuda Sync Ok" << std::endl;
        
        }
  
        {     
   
            eglrender::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = eglrender::BufferImageFormat::UNSIGNED_BYTE4;

            std::cerr << "Buffer Output  Ok" << std::endl;
       
//            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, output_buffer.getHostPointer());


        // Save the buffer as a PNG file
 //       saveAsPng("output.png", width, height, output_buffer.getHostPointer());
          eglrender::saveImage(outfile.c_str(), buffer, false );
          
          std::cerr << "EGL Save Image Ok" << std::endl;
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
            
        // Cleanup
//        eglDestroySurface(display, surface);        
//        eglDestroyContext(display, context);        
//        eglTerminate(display);

        // Clean up resources
//        destroyEGLContext(EGLContext);
          printf("EGL context and surface successfully created and cleaned up.\n");  
        }
    }
    catch( std::exception& e )
    {       std::cerr << "Caught exception: " << e.what() << "\n";        return 1;
    }    return 0;
}
