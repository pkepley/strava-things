import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ActivityAnimator:
    def __init__( self, dfs, names, stride=5, trail=30, figsize=(8,4.5),
                  facecolor='k', fontcolor='w', df_route=None, routewidth=5,
                  routealpha=0.1, routecolor='w', linecolors=None):

        self.dfs = dfs
        self.names = names
        self.stride = stride
        self.trail = trail
        self.facecolor = facecolor
        self.fontcolor = fontcolor
        self.df_route = df_route
        self.routealpha = routealpha
        self.routecolor = routecolor
        self.routewidth = routewidth

        self.xmin = min([df.longitude.min() for df in dfs])
        self.xmax = max([df.longitude.max() for df in dfs])
        self.ymin = min([df.latitude.min() for df in dfs])
        self.ymax = max([df.latitude.max() for df in dfs])
        xw = self.xmax - self.xmin
        yw = self.ymax - self.ymin
        self.xmin = self.xmin - 0.05 * xw
        self.xmax = self.xmax + 0.05 * xw
        self.ymin = self.ymin - 0.05 * yw
        self.ymax = self.ymax + 0.05 * yw
        self.nt = max([df.index.max() for df in dfs]) // self.stride + 1

        self.figsize=figsize
        fig, ax = plt.subplots(facecolor=self.facecolor, figsize=self.figsize)
        self.fig = fig
        ax.set_aspect('equal')
        ax.set_facecolor(self.facecolor)
        ax.set_frame_on(False)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if self.df_route is not None:
            ax.plot(self.df_route['longitude'], self.df_route['latitude'],
                    alpha=self.routealpha, linewidth=self.routewidth,
                    c=self.routecolor)
        self.ax = ax

        if linecolors is not None:
            self.lines = [ax.plot([], [], c=c)[0] for c in linecolors]
        else:
            self.lines = [ax.plot([], [])[0] for _ in dfs]

    def anim_init(self):
        for line in self.lines:
            line.set_data([], [])

        return self.lines

    def anim_update(self, frame):
        frame_cutoff = self.stride * frame

        for lnum, line in enumerate(self.lines):
            frame_hi = min(self.dfs[lnum].index.max(), frame_cutoff)
            frame_lo = max(frame_cutoff - self.trail, 0)

            if frame_lo <= self.dfs[lnum].index.max():
                line.set_data(
                    self.dfs[lnum].longitude.loc[frame_lo:frame_hi],
                    self.dfs[lnum].latitude.loc[frame_lo:frame_hi]
                )

            else:
                line.set_data([], [])

        return self.lines
