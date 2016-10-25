VERSION = 0.9.16

.PHONY: all
all: vlfeat/libvl-x86.so vlfeat/libvl-x64.so

vlfeat-$(VERSION)-bin.tar.gz:
	wget -O vlfeat-$(VERSION)-bin.tar.gz.tmp "http://www.vlfeat.org/download/vlfeat-$(VERSION)-bin.tar.gz"
	test $$(md5sum vlfeat-$(VERSION)-bin.tar.gz.tmp | cut -d' ' -f1) = "ad8cc60a539b71c4994872d2793f8b2b"
	mv vlfeat-$(VERSION)-bin.tar.gz{.tmp,}

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
