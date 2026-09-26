Title: median() is wrong for lists with an even number of items

`median([4, 1, 3, 2])` returns `3`, but the median of 1, 2, 3, 4 is `2.5`
(the average of the two middle values). Odd-length lists work fine.
