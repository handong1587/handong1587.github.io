## Using objcopy to embed data file in an executable:

**Embedding a File in an Executable, aka Hello World, Version 5967**

[http://www.linuxjournal.com/content/embedding-file-executable-aka-hello-world-version-5967](http://www.linuxjournal.com/content/embedding-file-executable-aka-hello-world-version-5967)

**Linking binary data**

[https://dvdhrm.wordpress.com/tag/objcopy/](https://dvdhrm.wordpress.com/tag/objcopy/)

On Ubuntu 14.04 LTS, 86_64:

```
objcopy --input binary \
    --output elf64-x86-64 \
    --binary-architecture i386:x86-64 \
    youdatafile youdatafile.o
```

## How to deal with shared library?

Use statifier or Ermine. Both tools take as input dynamically linked executable 
and as output create self-contained executable with all shared libraries embedded.

[http://statifier.sourceforge.net/](http://statifier.sourceforge.net/)

[http://www.magicermine.com/](http://www.magicermine.com/)

```
/path/to/ErmineLightTrial.x86_64 you_executable_file \
    --ld_library_path=/path/to/ld_lib/ \
    --output=new_executable_file
```
