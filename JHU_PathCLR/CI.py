
def compute_CI(nums):
  avg = sum(nums) / len(nums)
  print(avg)

  s = 0
  for i in nums:
    s += (i - avg) * (i - avg)

  s /= (len(nums) - 1)
  s = s ** 0.5
  e = (1.96 * s) / (len(nums) ** 0.5)
  print("CI: ", avg, " + ", e)
