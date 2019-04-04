/* global data needs to be initialized somewhere */
void __call_constructors() {
	extern void (*__init_array_start)();
	extern void (*__init_array_end)();
	for (void (*(*p))() = &__init_array_start; p < &__init_array_end; ++p) {
		(*p)();
	}
}

void __call_destructors() {
	extern void (*__fini_array_start)();
	extern void (*__fini_array_end)();
	for (void (*(*p))() = &__fini_array_start; p < &__fini_array_end; ++p) {
		(*p)();
	}
}
