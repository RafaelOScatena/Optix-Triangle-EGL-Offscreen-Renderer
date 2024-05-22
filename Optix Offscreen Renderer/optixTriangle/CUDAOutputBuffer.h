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

#pragma once

#include <glad2/glad_egl.h> // Needs to be included before gl_interop

#include <Exception.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>

//#include "sutilapi.h"
#include "sampleConfig.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdlib>
#include <chrono>
#include <vector>


#define STB_IMAGE_IMPLEMENTATION
#include <tinygltf/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>

#include <fstream>



namespace eglrender
{


enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data =      nullptr;
    unsigned int      width = 0;
    unsigned int      height = 0;
    BufferImageFormat pixel_format;
    // The memory backed by data isn't always owned by ImageBuffer (e.g. in the case of
    // loadImage), so you can't always free the memory in a destructor. Additionally you
    // can't simply delete the memory in the client either, because on some systems the
    // heap isn't shared between the sutil library and the client. In this case you should
    // call destroy to free the memory.
     void destroy();
};




static float toSRGB( float c )
{
    float invGamma = 1.0f / 2.4f;
    float powed    = std::pow( c, invGamma );
    return c < 0.0031308f ? 12.92f * c : 1.055f * powed - 0.055f;
}


static void savePPM( const unsigned char* Pix, const char* fname, int wid, int hgt, int chan )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
        throw Exception( "savePPM: Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw Exception( "savePPM: Attempting to save image with channel count != 1, 3, or 4." );

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
        throw Exception( "savePPM: Could not open file for" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ( ( chan == 1 ? ( is_float ? 'Z' : '5' ) : ( chan == 3 ? ( is_float ? '7' : '6' ) : '8' ) ) )
            << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<unsigned char*>( Pix ) ), wid*hgt*chan*( is_float ? 4 : 1 ) );
    OutFile.close();
}

void saveImage( const char* fname, const ImageBuffer& image, bool disable_srgb_conversion )
{
    const std::string filename( fname );
    if( filename.length() < 5 )
        throw Exception( "sutil::saveImage(): Failed to determine filename extension" );

    const std::string ext = filename.substr( filename.length()-3 );
    if( ext == "PPM" || ext == "ppm" )
    {
        //
        // Note -- we are flipping image vertically as we write it into output buffer
        //
        const int32_t width  = image.width;
        const int32_t height = image.height;
        std::vector<unsigned char> pix( width*height*3 );

        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        pix[ dst_idx+0] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+0 ];
                        pix[ dst_idx+1] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+1 ];
                        pix[ dst_idx+2] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+2 ];
                    }
                }
            } break;

            case BufferImageFormat::FLOAT3:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 3*width*j            + 3*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            case BufferImageFormat::FLOAT4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            default:
            {
                throw Exception( "sutil::saveImage(): Unrecognized image buffer pixel format.\n" );
            }
        }

        savePPM( pix.data(), filename.c_str(), width, height, 3 );
    }

    else if(  ext == "PNG" || ext == "png" )
    {
        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                stbi_flip_vertically_on_write( true );
                if( !stbi_write_png(
                            filename.c_str(),
                            image.width,
                            image.height,
                            4, // components,
                            image.data,
                            image.width*sizeof( uchar4 ) //stride_in_bytes
                            ) )
                    throw Exception( "sutil::saveImage(): stbi_write_png failed" );
            } break;

            case BufferImageFormat::FLOAT3:
            {
                throw Exception( "sutil::saveImage(): saving of float3 images to PNG not implemented yet" );
            }

            case BufferImageFormat::FLOAT4:
            {
                throw Exception( "sutil::saveImage(): saving of float4 images to PNG not implemented yet" );
            }

            default:
            {
                throw Exception( "sutil::saveImage: Unrecognized image buffer pixel format.\n" );
            }
        }
    }

    else if(  ext == "EXR" || ext == "exr" )
    {
        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                throw Exception( "sutil::saveImage(): saving of uchar4 images to EXR not implemented yet" );
            }

            case BufferImageFormat::FLOAT3:
            {
                const char* err;
                int32_t ret = SaveEXR(
                        reinterpret_cast<float*>( image.data ),
                        image.width,
                        image.height,
                        3, // num components
                        static_cast<int32_t>( true ), // save_as_fp16
                        filename.c_str(),
                        &err );

                if( ret != TINYEXR_SUCCESS )
                    throw Exception( ( "sutil::saveImage( exr ) error: " + std::string( err ) ).c_str() );

            } break;

            case BufferImageFormat::FLOAT4:
            {
                const char* err;
                int32_t ret = SaveEXR(
                        reinterpret_cast<float*>( image.data ),
                        image.width,
                        image.height,
                        4, // num components
                        static_cast<int32_t>( true ), // save_as_fp16
                        filename.c_str(),
                        &err );

                if( ret != TINYEXR_SUCCESS )
                    throw Exception( ( "sutil::saveImage( exr ) error: " + std::string( err ) ).c_str() );
            } break;

            default:
            {
                throw Exception( "sutil::saveImage: Unrecognized image buffer pixel format.\n" );
            }
        }
    }
    else
    {
        throw Exception( ( "sutil::saveImage(): Failed unsupported filetype '" + ext + "'" ).c_str() );
    }
}

void ensureMinimumSize( int& w, int& h )
{
    if( w <= 0 )
        w = 1;
    if( h <= 0 )
        h = 1;
}

void ensureMinimumSize( unsigned& w, unsigned& h )
{
    if( w == 0 )
        w = 1;
    if( h == 0 )
        h = 1;
}



enum class CUDAOutputBufferType
{
    CUDA_DEVICE = 0, // not preferred, typically slower than ZERO_COPY
    GL_INTEROP  = 1, // single device only, preferred for single device
    ZERO_COPY   = 2, // general case, preferred for multi-gpu if not fully nvlink connected
    CUDA_P2P    = 3  // fully connected only, preferred for fully nvlink connected
};


template <typename PIXEL_FORMAT>
class CUDAOutputBuffer
{
public:
    CUDAOutputBuffer( CUDAOutputBufferType type, int32_t width, int32_t height );
    ~CUDAOutputBuffer();

    void setDevice( int32_t device_idx ) { m_device_idx = device_idx; }
    void setStream( CUstream stream    ) { m_stream     = stream;     }

    void resize( int32_t width, int32_t height );

    // Allocate or update device pointer as necessary for CUDA access
    PIXEL_FORMAT* map();
    void unmap();

    int32_t        width() const  { return m_width;  }
    int32_t        height() const { return m_height; }

    // Get output buffer
    GLuint         getPBO();
    void           deletePBO();
    PIXEL_FORMAT*  getHostPointer();

private:
    void makeCurrent() { CUDA_CHECK( cudaSetDevice( m_device_idx ) ); }

    CUDAOutputBufferType       m_type;

    int32_t                    m_width             = 0u;
    int32_t                    m_height            = 0u;

    cudaGraphicsResource*      m_cuda_gfx_resource = nullptr;
    GLuint                     m_pbo               = 0u;
    PIXEL_FORMAT*              m_device_pixels     = nullptr;
    PIXEL_FORMAT*              m_host_zcopy_pixels = nullptr;
    std::vector<PIXEL_FORMAT>  m_host_pixels;

    CUstream                   m_stream            = 0u;
    int32_t                    m_device_idx        = 0;
};


template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer( CUDAOutputBufferType type, int32_t width, int32_t height )
    : m_type( type )
{
    // Output dimensions must be at least 1 in both x and y to avoid an error
    // with cudaMalloc.
#if 0
    if( width < 1 || height < 1 )
    {
        throw sutil::Exception( "CUDAOutputBuffer dimensions must be at least 1 in both x and y." );
    }
#else
    ensureMinimumSize( width, height );
#endif

    // If using GL Interop, expect that the active device is also the display device.
    if( type == CUDAOutputBufferType::GL_INTEROP )
    {
        int current_device, is_display_device;
        CUDA_CHECK( cudaGetDevice( &current_device ) );
        CUDA_CHECK( cudaDeviceGetAttribute( &is_display_device, cudaDevAttrKernelExecTimeout, current_device ) );
        if( !is_display_device )
        {
            std::cerr << "GL interop is only available on display device, please use display device for optimal "
             "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
             "degraded performance."
          << std::endl;

// Optionally, you can handle the error in other ways, like setting an error code, exiting the program, etc.
// For example:


        }
    }
    resize( width, height );
}


template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer()
{
    try
    {
        makeCurrent();
        if( m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P )
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_device_pixels ) ) );
        }
        else if( m_type == CUDAOutputBufferType::ZERO_COPY )
        {
            CUDA_CHECK( cudaFreeHost( reinterpret_cast<void*>( m_host_zcopy_pixels ) ) );
        }
        else if( m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P )
        {
            CUDA_CHECK( cudaGraphicsUnregisterResource( m_cuda_gfx_resource ) );
        }

        if( m_pbo != 0u )
        {
            GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
            GL_CHECK( glDeleteBuffers( 1, &m_pbo ) );
        }
    }
    catch(std::exception& e )
    {
        std::cerr << "CUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
    }
}


template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::resize( int32_t width, int32_t height )
{
    // Output dimensions must be at least 1 in both x and y to avoid an error
    // with cudaMalloc.
    ensureMinimumSize( width, height );

    if( m_width == width && m_height == height )
        return;

    m_width  = width;
    m_height = height;

    makeCurrent();

    if( m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_device_pixels ) ) );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_device_pixels ),
                    m_width*m_height*sizeof(PIXEL_FORMAT)
                    ) );

    }

    if( m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P )
    {
        // GL buffer gets resized below
        GL_CHECK( glGenBuffers( 1, &m_pbo ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData( GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT)*m_width*m_height, nullptr, GL_STREAM_DRAW ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0u ) );

        CUDA_CHECK( cudaGraphicsGLRegisterBuffer(
                    &m_cuda_gfx_resource,
                    m_pbo,
                    cudaGraphicsMapFlagsWriteDiscard
                    ) );
    }

    if( m_type == CUDAOutputBufferType::ZERO_COPY )
    {
        CUDA_CHECK( cudaFreeHost( reinterpret_cast<void*>( m_host_zcopy_pixels ) ) );
        CUDA_CHECK( cudaHostAlloc(
                    reinterpret_cast<void**>( &m_host_zcopy_pixels ),
                    m_width*m_height*sizeof(PIXEL_FORMAT),
                    cudaHostAllocPortable | cudaHostAllocMapped
                    ) );
        CUDA_CHECK( cudaHostGetDevicePointer(
                    reinterpret_cast<void**>( &m_device_pixels ),
                    reinterpret_cast<void*>( m_host_zcopy_pixels ),
                    0 /*flags*/
                    ) );
    }

    if( m_type != CUDAOutputBufferType::GL_INTEROP && m_type != CUDAOutputBufferType::CUDA_P2P && m_pbo != 0u )
    {
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData( GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT)*m_width*m_height, nullptr, GL_STREAM_DRAW ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0u ) );
    }

    if( !m_host_pixels.empty() )
        m_host_pixels.resize( m_width*m_height );
}


template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::map()
{
    if( m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P )
    {
        // nothing needed
    }
    else if( m_type == CUDAOutputBufferType::GL_INTEROP  )
    {
        makeCurrent();

        size_t buffer_size = 0u;
        CUDA_CHECK( cudaGraphicsMapResources ( 1, &m_cuda_gfx_resource, m_stream ) );
        CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(
                    reinterpret_cast<void**>( &m_device_pixels ),
                    &buffer_size,
                    m_cuda_gfx_resource
                    ) );
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        // nothing needed
    }

    return m_device_pixels;
}


template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::unmap()
{
    makeCurrent();

    if( m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P )
    {
        CUDA_CHECK( cudaStreamSynchronize( m_stream ) );
    }
    else if( m_type == CUDAOutputBufferType::GL_INTEROP  )
    {
        CUDA_CHECK( cudaGraphicsUnmapResources ( 1, &m_cuda_gfx_resource,  m_stream ) );
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        CUDA_CHECK( cudaStreamSynchronize( m_stream ) );
    }
}


template <typename PIXEL_FORMAT>
GLuint CUDAOutputBuffer<PIXEL_FORMAT>::getPBO()
{
    if( m_pbo == 0u )
        GL_CHECK( glGenBuffers( 1, &m_pbo ) );

    const size_t buffer_size = m_width*m_height*sizeof(PIXEL_FORMAT);

    if( m_type == CUDAOutputBufferType::CUDA_DEVICE )
    {
        // We need a host buffer to act as a way-station
        if( m_host_pixels.empty() )
            m_host_pixels.resize( m_width*m_height );

        makeCurrent();
        CUDA_CHECK( cudaMemcpy(
                    static_cast<void*>( m_host_pixels.data() ),
                    m_device_pixels,
                    buffer_size,
                    cudaMemcpyDeviceToHost
                    ) );

        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData(
                    GL_ARRAY_BUFFER,
                    buffer_size,
                    static_cast<void*>( m_host_pixels.data() ),
                    GL_STREAM_DRAW
                    ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    }
    else if( m_type == CUDAOutputBufferType::GL_INTEROP  )
    {
        // Nothing needed
    }
    else if ( m_type == CUDAOutputBufferType::CUDA_P2P )
    {
        makeCurrent();
        void* pbo_buff = nullptr;
        size_t dummy_size = 0;

        CUDA_CHECK( cudaGraphicsMapResources( 1, &m_cuda_gfx_resource, m_stream ) );
        CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( &pbo_buff, &dummy_size, m_cuda_gfx_resource ) );
        CUDA_CHECK( cudaMemcpy( pbo_buff, m_device_pixels, buffer_size, cudaMemcpyDeviceToDevice ) );
        CUDA_CHECK( cudaGraphicsUnmapResources( 1, &m_cuda_gfx_resource, m_stream ) );
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, m_pbo ) );
        GL_CHECK( glBufferData(
                    GL_ARRAY_BUFFER,
                    buffer_size,
                    static_cast<void*>( m_host_zcopy_pixels ),
                    GL_STREAM_DRAW
                    ) );
        GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    }

    return m_pbo;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::deletePBO()
{
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    GL_CHECK( glDeleteBuffers( 1, &m_pbo ) );
    m_pbo = 0;
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer()
{
    if( m_type == CUDAOutputBufferType::CUDA_DEVICE ||
        m_type == CUDAOutputBufferType::CUDA_P2P ||
        m_type == CUDAOutputBufferType::GL_INTEROP  )
    {
        m_host_pixels.resize( m_width*m_height );

        makeCurrent();
        CUDA_CHECK( cudaMemcpy(
                    static_cast<void*>( m_host_pixels.data() ),
                    map(),
                    m_width*m_height*sizeof(PIXEL_FORMAT),
                    cudaMemcpyDeviceToHost
                    ) );
        unmap();

        return m_host_pixels.data();
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        return m_host_zcopy_pixels;
    }
}

} // end namespace sutil
