/* Research-only CPU adapter: inspect_only deliberately leaves the engine's
 * tokenizer unloaded. Compile the unchanged core into this research object,
 * then load just its existing vocabulary; never create an inference graph. */
#ifndef DS4_NO_GPU
#error "generated inspect adapter must be compiled without GPU support"
#endif
#include "../ds4.c"
#include "mellum_generate_driver.h"
int mellum_generated_inspect_open(ds4_engine **out,const ds4_engine_options *options) {
    if(!out || !options || !options->inspect_only || options->backend!=DS4_BACKEND_CPU) return -1;
    int rc=ds4_engine_open(out,options);
    if(rc) return rc;
    if(!ds4_model_is_mellum()) { ds4_engine_close(*out);*out=NULL;return -1; }
    vocab_load(&(*out)->vocab,&(*out)->model);
    return 0;
}
