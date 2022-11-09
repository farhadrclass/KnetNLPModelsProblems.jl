# Set display width, load packages, import symbols
ENV["COLUMNS"]=72
using Base.Iterators: take, drop, cycle, Stateful
using IterTools: ncycle, takenth, takewhile
using MLDatasets: MNIST
using Knet
using Random


# Load MNIST data as an iterator of (x,y) minibatches
xtst,ytst = MNIST.testdata(Float32)
dtst = minibatch(xtst, ytst, 100)

summary.(first(dtst))

A = first(dtst)

next = iterate(dtst)
while next != nothing
  x, state = next
  # do something with x
  println(summary.(first(dtst)))
  println(summary(x))
  println(state)
  next = iterate(dtst, state)
#   break
end