#ifndef _SIGNALPROC_H_
#define _SIGNALPROC_H_
#include "chunkedArray.h"
#include "definitions.h"

/*!
*
*
*    \class SignalProc
*
*    \brief This class contains the preprocessing and past processing methods to perform real transform using complex transform \n
       to be able to use same amount of memory for all the possible transfmroation, we utilize the relation between DCT10 (DCT01) \n
*     and DST10 to calculate DST10 (DST01).
*
**/

class SignalProc
{
    public:
    SignalProc(); /*! class constructor */

#pragma acc routine vector
    void postprocessSignalDCT10( ChunkedArray &P, const int size, const int i,
                                 const int direction ); /*!< Discrete Cosine Transformation Post Process DCT10 */
#pragma acc routine vector
    void postprocessSignalDCT01( ChunkedArray &P, const int size, const int j,
                                 const int direction ); /*!< Discrete Cosine Transformation Post Process DCT01 */

#pragma acc routine vector
    void preprocessSignalDCT01( ChunkedArray &P, const int size, const int i,
                                const int direction ); /*!< preparing the signal for DCT01 tranmsform   */

#pragma acc routine vector
    void preprocessSignalDCT10( ChunkedArray &P, const int size, const int i,
                                const int direction ); /*!< preparing the signal for DCT10 tranmsform   */

#pragma acc routine vector
    void preprocessSignalDST10( ChunkedArray &P, const int size, const int i,
                                const int direction ); /*!< preparing the signal for DST10 tranmsform*/

#pragma acc routine vector
    void postprocessSignalDST10( ChunkedArray &P, const int size, const int i,
                                 const int direction ); /*!< postprocessing to obtain DST10 tranmsform*/

#pragma acc routine vector
    void preprocessSignalDST01( ChunkedArray &P, const int size, const int i,
                                const int direction ); /*!< preparing the signal for DST01 tranmsform*/

#pragma acc routine vector
    void postprocessSignalDST01( ChunkedArray &P, const int size, const int i,
                                 const int direction ); /*!< postprocessing to obtain DST01 tranmsform*/

#pragma acc routine vector
    void swap( ChunkedArray &P, const int size, const int i, const int direction ); /*!< reorganizes the elements for transformation*/

    void copyin(); /*!<copies  'this' object of the class to the GPU */

    ~SignalProc(); /*!< Class destructor*/
};

#endif
