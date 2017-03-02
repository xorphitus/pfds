data Queue a = Q [a] [a]

pushBack :: Queue a -> a -> Queue a
pushBack (Q xs ys) x = Q xs (x:ys)

popFront :: Queue a -> (a, Queue a)
popFront (Q []     ys) = popFront (Q (reverse ys) [])
popFront (Q (x:xs) ys) = (x, Q xs ys)

main :: IO ()
main = do
  putStr (fst (popFront q))
  where q = popFront (pushBack (Q [] []) "a")
