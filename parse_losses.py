import re

FILE = "vae/runs/celeba_esm_20260606_105940/losses.log"
losses = []

with open(FILE, 'r') as file:
	for line in file:
		matches = re.findall(r'step (\d+) | total: (\S+) | recon: (\S+) | percep: (\S+) | gan: (\S+) | reg: (\S+)', line.strip())
		print(matches)
		losses.append({
			"step": matches[0][0],
			"total": matches[1][1],
			"recon": matches[2][2],
			"percep": matches[3][3],
			"gan": matches[4][4],
			"reg": matches[5][5]
		})

print(losses)
