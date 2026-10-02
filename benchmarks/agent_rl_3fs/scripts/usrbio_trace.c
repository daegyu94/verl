#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <hf3fs_usrbio.h>
static void *resolve(const char *name) {
 void *fn=dlsym(RTLD_NEXT,name);
 if(!fn) { void *h=dlopen("libhf3fs_api_shared.so",RTLD_LAZY|RTLD_NOLOAD); if(h) fn=dlsym(h,name); }
 if(!fn) { fprintf(stderr,"usrbio trace: cannot resolve %s\n",name); _exit(125); }
 return fn;
}
static uint64_t ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static void log_event(const char *op,int fd,uint64_t off,uint64_t len,int ret,uint64_t elapsed) {
 const char *dir=getenv("AGENT_RL_USRBIO_TRACE"); if(!dir) return;
 char file[4096],buf[8192],path[4096]="",link[64];
 snprintf(file,sizeof(file),"%s/usrbio-%d.jsonl",dir,getpid());
 if(fd>=0) { snprintf(link,sizeof(link),"/proc/self/fd/%d",fd); ssize_t n=readlink(link,path,sizeof(path)-1); if(n>=0) path[n]=0; }
 int n=snprintf(buf,sizeof(buf),"{\"t_ns\":%lu,\"pid\":%d,\"tid\":%ld,\"op\":\"%s\",\"fd\":%d,\"path\":\"%s\",\"offset\":%lu,\"bytes\":%lu,\"ret\":%d,\"duration_ns\":%lu}\n",ns(),getpid(),syscall(SYS_gettid),op,fd,path,off,len,ret,elapsed);
 FILE *f=fopen(file,"a"); if(f) { fwrite(buf,1,n,f); fclose(f); }
}
int hf3fs_prep_io(const struct hf3fs_ior *ior,const struct hf3fs_iov *iov,bool read,void *ptr,int fd,size_t off,uint64_t len,const void *userdata) {
 static int(*real)(const struct hf3fs_ior*,const struct hf3fs_iov*,bool,void*,int,size_t,uint64_t,const void*);
 if(!real) real=resolve("hf3fs_prep_io"); uint64_t start=ns(); int ret=real(ior,iov,read,ptr,fd,off,len,userdata);
 log_event(read?"read":"write",fd,off,len,ret,ns()-start);return ret;
}
int hf3fs_wait_for_ios(const struct hf3fs_ior *ior,struct hf3fs_cqe *cqes,int cqec,int min_results,const struct timespec *timeout) {
 static int(*real)(const struct hf3fs_ior*,struct hf3fs_cqe*,int,int,const struct timespec*);
 if(!real) real=resolve("hf3fs_wait_for_ios"); uint64_t start=ns();int ret=real(ior,cqes,cqec,min_results,timeout);uint64_t bytes=0;
 if(ret>0) for(int i=0;i<ret;i++) if(cqes[i].result>0) bytes+=cqes[i].result;
 log_event(ior->for_read?"read_wait":"write_wait",-1,0,bytes,ret,ns()-start);return ret;
}
int hf3fs_reg_fd(int fd,uint64_t flags) {
 static int(*real)(int,uint64_t);if(!real) real=resolve("hf3fs_reg_fd");uint64_t start=ns();int ret=real(fd,flags);log_event("register",fd,0,0,ret,ns()-start);return ret;
}
