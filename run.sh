# 判断build文件夹是否存在，不存在则创建
if [ ! -d "/build" ]; then
  mkdir /build
fi

# 进入build
cd build

# 编译并运行
make clean
cmake ..
make
./smooth
