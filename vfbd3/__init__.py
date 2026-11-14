"""
vfbd3 - video restyling with diffusion models.

Two engines share the same video I/O, control extraction, motion estimation and resumable
work directory:

    vace   chunked generation with a video diffusion model (Wan VACE); temporal consistency
           comes from the model, and from feeding the tail of each chunk into the next.
    frame  per-frame img2img with motion-compensated feedback of the previous output, the
           technique from the original controlnetvideo.
"""

__version__ = "0.1.0"
