VERSION = 0.9.20
CHECKSUM = e22ada7ddd708d739ed9958b16642ba1

.PHONY: all
all: vlfeat/libvl-x86.so vlfeat/libvl-x64.so

vlfeat-$(VERSION)-bin.tar.gz:
	wget -O vlfeat-$(VERSION)-bin.tar.gz.tmp "http://www.vlfeat.org/download/vlfeat-$(VERSION)-bin.tar.gz"
	test $$(md5sum vlfeat-$(VERSION)-bin.tar.gz.tmp | cut -d' ' -f1) = "$(CHECKSUM)"
	mv vlfeat-$(VERSION)-bin.tar.gz{.tmp,}
	@touch "$@"

vlfeat/libvl-x86.so: vlfeat-$(VERSION)-bin.tar.gz
	tar -C vlfeat -xz -f "$<" vlfeat-$(VERSION)/bin/glnx86/libvl.so \
		--strip-components=3 --transform 's/libvl.so/libvl-x86.so/'
	@touch "$@"

vlfeat/libvl-x64.so: vlfeat-$(VERSION)-bin.tar.gz
	tar -C vlfeat -xz -f "$<" vlfeat-$(VERSION)/bin/glnxa64/libvl.so \
		--strip-components=3 --transform 's/libvl.so/libvl-x64.so/'
	@touch "$@"

.PHONY: clean
clean:
	rm -f vlfeat-$(VERSION)-bin.tar.gz
	rm -f vlfeat/libvl-*.so
	rm -f vlfeat/*.pyc
