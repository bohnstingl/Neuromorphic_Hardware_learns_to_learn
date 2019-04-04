#pragma once

#include "libnux/attrib.h"

ATTRIB_UNUSED static void sync() {
	asm volatile("sync");
}
